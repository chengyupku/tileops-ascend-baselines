"""Ascend NPU cumulative-sum forward kernel: ``devices={"npu"}``, outside the CUDA kernel
hierarchy.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``CumsumFwdOp``; see
``docs/ascend950/dev/CumsumFwdOp/round_1_schedule.md`` for the full tiling design this file
implements. Registers itself under the ``"ascend"`` dispatch key
(``tileops.backend.register_kernel_type``), so importing this module is enough to make
``CumsumFwdOp`` route NPU tensors here -- no change to the existing CUDA
``CumulativeRowScanKernel``/``CumsumParallelScanKernel``/``CumulativeKernel`` or ``CumsumFwdOp``.

This is the project's first reduction/scan-family op: unlike batch 1's embarrassingly-parallel
elementwise kernels, a cumulative sum has a genuine sequential dependency along the scan axis.

**Why this is hand-built rather than calling the high-level scan builtin**: ``T.cumsum`` /
``T.cummax`` are syntactically callable on the Ascend dialect (re-exported from
``tilelang.language.common``) but have no registered lowering for the Ascend target --
``src/op/scan.cc``'s ``ResolveScanImpl`` only registers CUDA/ROCm implementations
(``src/cuda/op/scan.cc`` / ``src/rocm/op/scan.cc``; no ``src/ascend/op/scan.cc`` exists), so
calling either on this backend hits a fatal ``ICHECK`` ("no cumsum implementation registered for
<target>") at target-resolution time. Confirmed by the round 1 Planner reading the source, and
independently re-confirmed by this file's own M3 run (see round_1_result.md for the verbatim
error). Ascend also has no user-callable warp-shuffle primitive, so the existing CUDA
``CumulativeRowScanKernel``'s ``T.shfl_up``-based cross-lane combine (its own docstring: chunk
scans -> warp-shuffle combine -> broadcast) cannot be ported as-is either.

Does *not* inherit ``_CumulativeKernelBase``/``CumulativeKernel``/``CumulativeRowScanKernel``/
``CumsumParallelScanKernel`` (``tileops/kernels/reduction/cumulative.py``): all are CUDA-only
(``supported_archs = [80, 86, 89, 90]`` SM versions, and ``forward()`` calls
``self._require_cuda``, which hardcodes ``tensor.is_cuda``).

**Design** (round_1_schedule.md SS3): every one of this round's 4 benchmarked cases is a "whole
row fits in one block" case -- the largest scan axis (32768 bf16 elements = 64 KiB) is
comfortably under the ~216 KiB SIMT UB budget -- so there is no cross-block carry anywhere here;
parallelism across independent rows comes entirely from spreading rows over ``num_blocks`` cores,
and parallelism within one row's scan from splitting it across ``P`` lanes inside one block. Each
block scans ``rows_per_iter`` whole rows per iteration with a three-phase local scan, mirroring
the CUDA kernel's phase structure (chunked local scan -> cross-lane combine -> broadcast) with
phase B's mechanism replaced:

  Phase A: each of ``P`` lanes walks its own ``N / P``-element chunk sequentially, accumulating
      in an fp32 fragment register, and writes the running (not just final) local inclusive scan
      back into the *same* UB slot it read from -- one rounding to storage dtype per element, not
      compounded across the scan (the module reuses one UB buffer for input, local scan and final
      output; see the buffer-reuse note below).
  Phase B: a tiny scalar fixup -- no ``T.Parallel``/``T.SimtVF`` -- turns each row's ``P`` lane
      totals into exclusive prefix offsets with a trace-time-unrolled Python loop and
      ``T.alloc_var(dtype, init=...)`` scalar state, reassigned by bare name
      (``running = running + ...``), not indexed with ``[0]`` -- M3 found that this
      ``init=``-constructed form evaluates to an already-dereferenced scalar in this repo's eager
      trace builder (``running[0]`` raised ``TypeError: 'BufferLoad' object is not
      subscriptable``), matching the existing CUDA ``_parallel_scan_carry_kernel``'s usage
      exactly rather than ``tile_schedule.py``'s ``T.alloc_var(dtype)``-without-``init`` +
      ``[0]``-indexed form (round_1_schedule.md's H4 risk, resolved -- see round_1_result.md).
  Phase C: broadcast-add each lane's fp32 offset back across its chunk, narrowing to the output
      dtype only at this final write.

No explicit barrier is used between the phases: this relies on the same compiler-ordered UB
handoff across separate ``T.SimtVF`` scopes that ``example_rmsnorm.py``'s working kernel already
depends on (write inside one ``T.SimtVF`` scope, read after it with no manual sync) --
round_1_schedule.md's H2 extends that 2-scope precedent to 3 (SimtVF -> scalar -> SimtVF).

**Buffer reuse**: a single ``alloc_shared`` buffer (``x_ub``) serves as the DMA-in destination,
Phase A's local-scan output, and Phase C's final output, written in place at each phase (every
phase reads then writes its own ``(r, p)`` slot, so no cross-lane hazard exists within one
``T.Parallel`` statement). This was adopted from the start (round_1_schedule.md SS3.5 flags it as
"a real optimization opportunity ... not a round-1 requirement" assuming two separate buffers) --
it both halves the UB footprint, which matters most for the fp32 ``llama-13b-hidden`` case, and
simplifies the kernel.

Accumulation is fp32 throughout (``code-style.md``'s overflow-prone-math rule, and a real
numerical necessity here -- a scan's error compounds over up to 32768 terms, unlike a one-shot
elementwise op); the storage dtype is only ever reconstructed when writing a UB slot back.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.kernel_base import Entry, Kernel, vector_aligned
from tileops.kernels.reduction._primitives import (
    restore_same_shape,
    rows_for_axes,
    torch_dtype_nbytes,
)
from tileops.kernels.reduction.call_spec import CumsumFwdInterface, CumulativeCall

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only off-Ascend
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendCumsumKernel"]

# SIMT UB budget per vector core, in bytes -- same figure/citation as the batch-1 Ascend kernels
# (ascend_baselines/TileKernels/tile_kernels/config.py:get_max_ub_per_vector_core, 216 KiB in
# SIMT mode). round_1_schedule.md SS3.1 flags this as another codebase's self-tuned constant, not
# yet calibrated against this project's own compiler -- a compiler UB-overflow error is ground
# truth over this figure, not the other way around.
_UB_BUDGET_BYTES = 216 * 1024
# Reserve subtracted from the UB budget before sizing the double-buffered ``x_ub``, covering
# ``totals_ub``/``offsets_ub`` (also double-buffered under pipelining), Phase A's fp32 fragment
# accumulator, and the compiler's own event-slot/fragment-spill overhead
# (round_1_schedule.md SS7 risk 4: "tight UB margin" flagged specifically for the fp32
# llama-13b-hidden case). Generous rather than exact: underfitting this just drops
# ``rows_per_iter`` one notch (still correct, slightly less parallel); overfitting it risks a
# real dynamic-shared-memory overflow the compiler reports as fatal.
_UB_SCRATCH_RESERVE_BYTES = 64 * 1024
# Hard Ascend SIMT ceiling (dev_process.md SS7.5, re-confirmed on real hardware in batch 1): a
# block's thread count must stay within this regardless of tile size.
_MAX_THREADS = 2048
# round_1_schedule.md SS3.5: round 1 keeps ``rows_per_iter * P`` within this bound, so Phase B's
# trace-time-unrolled instruction count (one add + two stores per (r, p)) stays bounded.
_MAX_ROWS_TIMES_P = 512
# P candidates (round_1_schedule.md SS3.1): the three values that divide every one of this
# round's benchmarked N (4096, 32768, 5120) exactly, so no masked/padded tail handling is needed
# for any of the 4 cases. A non-benchmarked N that divides none of these falls back to
# ``_largest_divisor_leq`` below, which is always correct (worst case P=1, a fully serial scan --
# the degenerate case of this same three-phase design, not a separate fallback kernel).
_P_CANDIDATES = (512, 256, 128)
# Fallback vector-core count if the live device query fails: the confirmed real count for this
# 950 SKU (dev_process.md SS7.5, torch_npu.npu.get_device_properties measurement). Only changes
# how many persistent blocks launch -- ``_largest_divisor_leq`` always returns a divisor of M, so
# correctness never depends on this number being right.
_FALLBACK_VECTOR_CORES = 72


def _num_vector_cores() -> int:
    """Real AIV (vector) core count of the current NPU device, re-queried each call.

    round_1_schedule.md SS3.4 / dev_process.md SS7.5: the project's own cached 72 should still be
    re-confirmed per-worker via the live driver query rather than trusted blindly.
    ``tilelang.carver.arch.driver.get_num_vector_cores`` is the documented programmatic source of
    truth (reads ``torch_npu.npu.get_device_properties(...).vector_core_num``), not the
    ``example_vecadd.py`` demo's hardcoded 64.
    """
    try:
        from tilelang.carver.arch.driver import get_num_vector_cores

        cores = get_num_vector_cores(device_id=0)
        if cores:
            return int(cores)
    except Exception:
        pass
    return _FALLBACK_VECTOR_CORES


def _largest_divisor_leq(n: int, cap: int) -> int:
    """The largest divisor of *n* that does not exceed *cap*; always >= 1 for n >= 1."""
    cap = max(1, min(n, cap))
    for candidate in range(cap, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1  # unreachable: 1 always divides n


def _pick_tiling(M: int, N: int, itemsize: int, num_cores: int) -> tuple[int, int, int]:
    """Return ``(num_blocks, rows_per_iter, P)`` for one ``AscendCumsumKernel`` build.

    Generalizes round_1_schedule.md SS3.4/SS3.5's per-case arithmetic from the 4 concrete
    benchmark cases to any ``(M, N)``, since ``CumsumFwdInterface`` promises a correct scan for
    whatever shape the manifest hands over, not just the benchmarked ones.

    ``num_blocks``: the largest divisor of ``M`` not exceeding the real core count, so every
        block gets exactly the same row count across iterations -- no per-block tail.
    ``rows_per_iter``: the largest divisor of ``M // num_blocks`` whose double-buffered UB
        footprint (one buffer, reused in place by Phase A/C) fits under
        ``_UB_BUDGET_BYTES - _UB_SCRATCH_RESERVE_BYTES``, capped so at least the smallest ``P``
        candidate still satisfies ``rows_per_iter * P <= _MAX_ROWS_TIMES_P``.
    ``P``: the largest value in ``_P_CANDIDATES`` dividing ``N`` exactly while keeping
        ``rows_per_iter * P <= _MAX_ROWS_TIMES_P``; falls back to the largest divisor of ``N``
        under the same product cap (and the SIMT thread cap) when no candidate divides ``N``.
    """
    num_blocks = _largest_divisor_leq(M, num_cores)
    rows_per_group = M // num_blocks
    budget = _UB_BUDGET_BYTES - _UB_SCRATCH_RESERVE_BYTES
    rows_cap = max(1, min(rows_per_group, _MAX_ROWS_TIMES_P // min(_P_CANDIDATES)))
    rows_per_iter = 1
    for candidate in range(rows_cap, 0, -1):
        if rows_per_group % candidate:
            continue
        iterations = rows_per_group // candidate
        stages = 2 if iterations > 1 else 1
        if candidate * N * itemsize * stages <= budget:
            rows_per_iter = candidate
            break
    P = 0
    for candidate in _P_CANDIDATES:
        if N % candidate == 0 and rows_per_iter * candidate <= _MAX_ROWS_TIMES_P:
            P = candidate
            break
    if P == 0:
        cap = max(1, min(N, _MAX_ROWS_TIMES_P // rows_per_iter, _MAX_THREADS // rows_per_iter))
        P = _largest_divisor_leq(N, cap)
    return num_blocks, rows_per_iter, P


if T is not None:

    @functools.lru_cache(maxsize=None)
    def _build_cumsum_kernel(
        N: int, dtype_str: str, num_blocks: int, rows_per_iter: int, iterations: int, P: int
    ):
        """Build the three-phase block-scan program for one ``(N, dtype, tiling)`` instance.

        See the module docstring for the phase breakdown. The two branches below (pipelined vs.
        not) are written out in full rather than factored into a shared helper: a plain Python
        function containing ``T.Parallel``/``with T.SimtVF(...)`` only rewrites correctly when
        it is itself a directly-traced ``@T.prim_func``/``@T.macro`` body (dev_process.md SS7.5's
        "ForFrame object is not iterable" pitfall), and round 1's time budget favors the
        duplicated-but-certain form over an unverified ``@T.macro`` refactor.

        The input/output shape is written as ``num_blocks * iterations * rows_per_iter`` rather
        than naming a separate ``M`` parameter: a name used only inside a nested
        ``@T.prim_func``'s type annotation, never in its body, is not reliably captured as a
        closure cell by TileLang's eager type-hint resolution and resolves to a ``NameError`` at
        trace time (dev_process.md SS7.5 pitfall, hit by ``SiluFwdOp`` round 1) -- every name
        used here is also read in the body below.
        """
        chunk = N // P
        threads = rows_per_iter * P
        assert threads <= _MAX_THREADS, f"threads={threads} exceeds the {_MAX_THREADS} SIMT cap"
        assert chunk * P == N, f"P={P} must divide N={N} exactly"

        @T.prim_func
        def main(
            X: T.Tensor((num_blocks * iterations * rows_per_iter, N), dtype_str),
            Y: T.Tensor((num_blocks * iterations * rows_per_iter, N), dtype_str),
        ):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((rows_per_iter, N), dtype_str)
                totals_ub = T.alloc_shared((rows_per_iter, P), "float32")
                offsets_ub = T.alloc_shared((rows_per_iter, P), "float32")

                if iterations > 1:
                    T.annotate_buffer_versions({x_ub: 2, totals_ub: 2, offsets_ub: 2})
                    for it in T.Pipelined(iterations, num_stages=2):
                        row0 = (it * num_blocks + bx) * rows_per_iter
                        T.copy(X[row0 : row0 + rows_per_iter, :], x_ub)

                        # Phase A: per-lane local inclusive scan, fp32 register accumulator.
                        with T.SimtVF(threads=threads):
                            acc = T.alloc_fragment((rows_per_iter, P), "float32")
                            for r, p in T.Parallel(rows_per_iter, P):
                                acc[r, p] = T.float32(0)
                            for c in range(chunk):
                                for r, p in T.Parallel(rows_per_iter, P):
                                    v = T.cast(x_ub[r, p * chunk + c], "float32")
                                    acc[r, p] = acc[r, p] + v
                                    x_ub[r, p * chunk + c] = T.cast(acc[r, p], dtype_str)
                            T.copy(acc, totals_ub)

                        # Phase B: scalar fixup -- exclusive prefix of the P lane totals.
                        for r in range(rows_per_iter):
                            running = T.alloc_var("float32", init=0.0)
                            for p in range(P):
                                offsets_ub[r, p] = running
                                running = running + totals_ub[r, p]

                        # Phase C: broadcast-add the offset; narrow to storage dtype here only.
                        with T.SimtVF(threads=threads):
                            for c in range(chunk):
                                for r, p in T.Parallel(rows_per_iter, P):
                                    v = (
                                        T.cast(x_ub[r, p * chunk + c], "float32")
                                        + offsets_ub[r, p]
                                    )
                                    x_ub[r, p * chunk + c] = T.cast(v, dtype_str)

                        T.copy(x_ub, Y[row0 : row0 + rows_per_iter, :])
                else:
                    row0 = bx * rows_per_iter
                    T.copy(X[row0 : row0 + rows_per_iter, :], x_ub)

                    with T.SimtVF(threads=threads):
                        acc = T.alloc_fragment((rows_per_iter, P), "float32")
                        for r, p in T.Parallel(rows_per_iter, P):
                            acc[r, p] = T.float32(0)
                        for c in range(chunk):
                            for r, p in T.Parallel(rows_per_iter, P):
                                v = T.cast(x_ub[r, p * chunk + c], "float32")
                                acc[r, p] = acc[r, p] + v
                                x_ub[r, p * chunk + c] = T.cast(acc[r, p], dtype_str)
                        T.copy(acc, totals_ub)

                    for r in range(rows_per_iter):
                        running = T.alloc_var("float32", init=0.0)
                        for p in range(P):
                            offsets_ub[r, p] = running
                            running = running + totals_ub[r, p]

                    with T.SimtVF(threads=threads):
                        for c in range(chunk):
                            for r, p in T.Parallel(rows_per_iter, P):
                                v = T.cast(x_ub[r, p * chunk + c], "float32") + offsets_ub[r, p]
                                x_ub[r, p * chunk + c] = T.cast(v, dtype_str)

                    T.copy(x_ub, Y[row0 : row0 + rows_per_iter, :])

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendCumsumKernel(Kernel, CumsumFwdInterface):
    """Ascend NPU inclusive prefix sum (cumsum) along a contiguous trailing axis.

    See ``docs/ascend950/dev/CumsumFwdOp/round_1_schedule.md`` for the three-phase block-scan
    design this implements (hand-built: ``T.cumsum`` has no registered Ascend lowering, see the
    module docstring). ``devices`` is NPU only; ``supported_archs`` is left ``None`` -- unlike
    the CUDA cumulative-kernel hierarchy this class does not inherit, there is no Ascend
    SoC-version gate here to replace the CUDA SM-version one.

    Accumulates in fp32 throughout; the output has the same shape and dtype as the input, and
    nothing is written in place (``CumsumFwdInterface``'s contract).

    Tolerance: checked against the shared reduction test tolerance
    (``workloads.reduction.reduction_verification``), which scales with ``sqrt(N) * rms(x)`` --
    the expected magnitude of a long random-walk prefix sum -- rather than a fixed epsilon.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: CumulativeCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: CumulativeCall) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendCumsumKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        if call.op_kind != "sum":
            raise ValueError(
                f"AscendCumsumKernel only serves op_kind='sum', got {call.op_kind!r}"
            )
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.M = call.m
        self.N = call.n
        self.dtype = call.dtype
        self.scan_axis = call.axis
        itemsize = torch_dtype_nbytes(self.dtype)
        num_cores = _num_vector_cores()
        num_blocks, rows_per_iter, P = _pick_tiling(self.M, self.N, itemsize, num_cores)
        iterations = (self.M // num_blocks) // rows_per_iter
        self.tiling = {
            "num_blocks": num_blocks,
            "rows_per_iter": rows_per_iter,
            "iterations": iterations,
            "P": P,
        }
        self.kernel = _build_cumsum_kernel(
            self.N, self.dtype_str, num_blocks, rows_per_iter, iterations, P
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan ``self.scan_axis`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``self.dtype`` on an NPU device.

        Returns:
            A new tensor shaped and typed like *x*.
        """
        if x.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {x.device}. "
                "Another target's backend serves other devices."
            )
        x = vector_aligned(x)
        in_shape = tuple(x.shape)
        axes = (self.scan_axis,)
        rows = rows_for_axes(x, axes)
        if tuple(rows.shape) != (self.M, self.N):
            raise ValueError(
                f"{type(self).__name__} was built for an ({self.M}, {self.N}) row layout, "
                f"got {tuple(rows.shape)}"
            )
        y = self.kernel(rows)
        return restore_same_shape(y, in_shape, axes)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("CumsumFwdOp", "ascend", AscendCumsumKernel)
except ImportError:
    pass
