"""Ascend NPU GELU forward kernel (exact-erf form, fp16/bf16 storage).

New Ascend-only implementation of ``GeluFwdOp``, registered under the dispatch key
``"ascend"`` (see ``docs/ascend950/dev_process.md`` §1) alongside the existing in-tree CUDA
keys ``"gelu"``/``"gelu_tanh"``. This module only *adds* an implementation -- it does not
import, edit, or otherwise touch
``tileops.kernels.elementwise.activations.GeluFwdKernel`` or ``ops/elementwise/activations.py``.

Round-1 design: ``docs/ascend950/dev/GeluFwdOp/round_1_schedule.md``. Per that schedule, the
3 benchmarked manifest cases are all fp16/bf16 storage and all use the default
``approximate="none"``, so this kernel refuses ``approximate="tanh"`` (that case falls through
to the unavailable CUDA ``gelu_tanh`` key, i.e. "no implementation serves this call", rather
than silently computing the wrong formula) and never reaches the ``T.erf`` branch of the
shared polynomial helper.

Round-2 design: ``docs/ascend950/dev/GeluFwdOp/round_2_schedule.md``. Round 1's decode case
(14,336 elements, ``T.Kernel(1)``, all work on one AIV core) missed its target by 2.02x;
round 2 replaces that single-core decode path with a 64-core grid (``_build_decode_multicore``,
224 elements/core), reusing the same ``_erf.gelu`` math unchanged. ``_build_single_shot`` is
kept as a defensive fallback for any future small shape that does not divide evenly by the
core count (see ``_mode_for``), but no longer serves this op's own decode case.

Importing this module on a process with no Ascend toolchain installed is a no-op: the
``tilelang.ascend`` import is wrapped in ``try/except ImportError``, so no class is defined
and no registration happens.
"""

import functools

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise._erf import gelu
from tileops.kernels.elementwise.call_spec import GeluCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGeluKernel"]

try:
    import tilelang
    import tilelang.ascend.language as T
except ImportError:  # pragma: no cover - exercised only where the Ascend toolchain is absent
    pass
else:
    # Tiling, round_1_schedule.md §3.3: anchored on the one proven-working Ascend elementwise
    # example (``tilelang/examples/ascend/example_vecadd.py``: num_blocks=64, tile=8192,
    # num_stages=2, num_threads=2048), not a cited hardware UB-capacity figure (none was found
    # in docs/ascend950/*.md or the tilelang-ascend skill docs this round).
    #
    # ``_NUM_BLOCKS`` (64) does double duty as of round 2: both the pipelined prefill grid
    # (below) and ``_build_decode_multicore``'s grid are sized to it -- both mean the same
    # thing, "one block per physical AIV core" (round_2_schedule.md §2 names this
    # ``_NUM_AIV_CORES``; reusing ``_NUM_BLOCKS`` here instead of adding a second
    # same-value constant, per code-style.md's "several functions must agree on it" rule).
    _NUM_BLOCKS = 64
    _TILE = 8192
    _NUM_STAGES = 2
    _THREADS = 2048
    # Both prefill cases (29,360,128 elements) divide this exactly (56 iterations, no tail).
    # The decode case (14,336 elements) is far smaller than one pipelined block's tile and
    # is instead split across all _NUM_BLOCKS cores by ``_build_decode_multicore`` below
    # (round_2_schedule.md §2 -- round 1's single-core ``_build_single_shot`` path left 63 of
    # 64 AIV cores idle and missed its target by 2.02x).

    # Ceiling for the single-shot (no-pipelining, single-core) fallback path: total GM<->UB
    # traffic at 4 bytes/elem (fp16/bf16 in + out) stays a small fraction of the ~192 KiB
    # envelope example_vecadd.py's own (3-buffer, fp32, double-buffered) footprint already
    # demonstrates works. Also used as the upper bound for the decode-multicore path's
    # per-call total (split across cores, so each core's actual footprint is far smaller).
    _SINGLE_SHOT_MAX_ELEMENTS = 65536

    def _mode_for(n_total: int) -> str:
        """Which builder serves *n_total*; defensive per round_1_schedule.md §5 risk 4.

        Order of preference for a small (non-pipelined) shape, per round_2_schedule.md §2:
        prefer spreading across all ``_NUM_BLOCKS`` AIV cores (``decode_multicore``) whenever
        the count divides evenly, since round 2 measured a 1-core decode kernel losing to an
        otherwise-heavier-workload 64-core sibling kernel on the identical element count.
        ``single_shot`` (one core, round 1's original path) remains as a fallback for a small
        shape that does *not* divide evenly by the core count -- kept rather than deleted so
        a future caller with an awkward small shape still has a correct (if unoptimized) path
        instead of a hard failure.
        """
        if n_total % (_NUM_BLOCKS * _TILE) == 0:
            return "pipelined"
        if n_total <= _SINGLE_SHOT_MAX_ELEMENTS and n_total % _NUM_BLOCKS == 0:
            return "decode_multicore"
        if n_total <= _SINGLE_SHOT_MAX_ELEMENTS:
            return "single_shot"
        raise NotImplementedError(
            f"AscendGeluKernel: n_total={n_total} divides neither into "
            f"{_NUM_BLOCKS}x{_TILE} pipelined blocks nor fits the "
            f"{_SINGLE_SHOT_MAX_ELEMENTS}-element single-shot/decode-multicore path. "
            "Round-1 tiling (docs/ascend950/dev/GeluFwdOp/round_1_schedule.md §3.3) covers "
            "only the 3 benchmarked manifest cases; a general tail handler was deliberately "
            "not built this round."
        )

    @functools.lru_cache(maxsize=None)
    def _build_pipelined(n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int):
        """Multi-block, ``T.Pipelined`` GELU kernel for the two prefill cases."""
        assert n_total % (num_blocks * tile) == 0, (
            f"_build_pipelined requires n_total={n_total} to divide evenly into "
            f"num_blocks={num_blocks} x tile={tile}"
        )
        iterations = n_total // (num_blocks * tile)
        threads = min(tile, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    with T.SimtVF(threads=threads):
                        for i in T.Parallel(tile):
                            # Promote to fp32 inside the per-lane body (code-style.md), not at
                            # the UB-buffer level: _erf.gelu casts to fp32 internally and
                            # returns an fp32 PrimExpr; cast back to storage dtype at the
                            # boundary.
                            y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_single_shot(n_total: int, dtype_str: str):
        """Single-block, no-pipelining GELU kernel; fallback for a small shape that does not
        divide evenly across ``_NUM_BLOCKS`` AIV cores.

        Round 1 used this path for the decode case too, but round 2 measured that running
        the whole shape on 1 of 64 AIV cores (leaving the rest idle) missed the decode
        target by 2.02x (round_2_schedule.md §1); ``_build_decode_multicore`` below now
        serves any shape this path previously handled that also divides evenly by
        ``_NUM_BLOCKS``. This single-core path is kept, unmodified, strictly as a defensive
        fallback (round_2_schedule.md §2 point 3) -- not deleted, just no longer on the
        decode case's dispatch path.
        """
        threads = min(n_total, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(n_total):
                        y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_decode_multicore(n_total: int, dtype_str: str, num_cores: int):
        """Grid-of-``num_cores`` GELU kernel for a small shape that divides evenly by
        ``num_cores``; round 2's fix for the decode case (round_2_schedule.md §2).

        Every core gets exactly one statically-sized ``block = n_total // num_cores``-element
        tile -- no re-looping, hence a plain ``T.Kernel(num_cores) as core_id:`` body rather
        than ``T.Persistent`` (round_2_schedule.md §2 point 2: with exactly one tile per
        core, ``T.Persistent``'s general re-looping machinery degenerates to exactly what
        this direct form already expresses, so the simpler form is used). Structurally this
        ports ``AscendGeluAndMulKernel``'s decode-case grid shape
        (``gelu_and_mul.py``'s ``_gelu_and_mul_program`` with ``m=1``), not its math -- the
        GELU polynomial itself is still the shared, unmodified ``_erf.gelu`` import.
        """
        assert n_total % num_cores == 0, (
            f"_build_decode_multicore requires n_total={n_total} to divide evenly by "
            f"num_cores={num_cores}; _mode_for should not route here otherwise"
        )
        block = n_total // num_cores
        threads = min(block, _THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_cores) as core_id:
                x_ub = T.alloc_shared((block,), dtype_str)
                y_ub = T.alloc_shared((block,), dtype_str)
                begin = core_id * block
                T.copy(X[begin : begin + block], x_ub)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(block):
                        y_ub[i] = T.cast(gelu(x_ub[i], dtype_str), dtype_str)
                T.copy(y_ub, Y[begin : begin + block])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    class AscendGeluKernel(Kernel, UnaryElementwiseFwdInterface):
        """Ascend NPU element-wise GELU, exact-erf form only.

        Mirrors the in-tree CUDA ``GeluFwdKernel``'s numerics exactly: fp16/bf16 output is
        evaluated by the same hand-rolled, always-fp32, clamped degree-10 Horner polynomial
        (``tileops.kernels.elementwise._erf.gelu``, imported unmodified -- not re-derived),
        whose worst-case error over the real line is 1.7e-5, an order below half a float16
        ulp at 1.0 and well inside this project's fp16 (1e-3) and bf16 (1.6e-2) tolerance
        table. Registered under the ``"ascend"`` dispatch key, not ``"gelu"``/``"gelu_tanh"``
        (already taken by the CUDA kernels on the same interface); selected automatically for
        NPU-device calls because ``devices = frozenset({"npu"})`` makes every CUDA key
        ``unavailable()`` there.
        """

        devices = frozenset({"npu"})

        def __init__(self, call: GeluCall, *, device_index: "int | None" = None) -> None:
            super().__init__(device_index=device_index)
            self.n_total = call.n_total
            self.dtype = call.dtype
            self.approximate = call.approximate
            dtype_str = self.dtype_to_str(call.dtype)
            mode = _mode_for(call.n_total)
            if mode == "pipelined":
                self.kernel = _build_pipelined(
                    call.n_total, dtype_str, _NUM_BLOCKS, _TILE, _NUM_STAGES
                )
            elif mode == "decode_multicore":
                self.kernel = _build_decode_multicore(call.n_total, dtype_str, _NUM_BLOCKS)
            else:
                self.kernel = _build_single_shot(call.n_total, dtype_str)

        @classmethod
        def refusal(cls, call: GeluCall) -> "str | None":
            if call.approximate != "none":
                return f"evaluates the erf form, not approximate={call.approximate!r}"
            return super().refusal(call)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Return a new tensor, GELU applied element-wise; ``input`` is left untouched."""
            flat = input.reshape(-1)
            result = self.kernel(flat)
            return result.reshape(input.shape)

    register_kernel_type("GeluFwdOp", "ascend", AscendGeluKernel)
