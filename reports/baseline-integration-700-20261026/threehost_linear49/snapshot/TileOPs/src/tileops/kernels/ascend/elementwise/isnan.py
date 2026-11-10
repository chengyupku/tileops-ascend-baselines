"""Ascend NPU IsNaN forward kernel: ``y = isnan(x)``, bool output, via the
``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for
``IsnanFwdOp``; see ``docs/ascend950/dev/IsnanFwdOp/round_1_schedule.md`` for the full
design this file implements (three real compile probes, no NPU touched, de-risked
``T.isnan`` availability and the asymmetric bool-output UB footprint before any of this
was written). Registers an additional implementation of ``IsnanFwdOp``'s own
``UnaryPredicateFwdInterface`` under the ``"ascend"`` dispatch key
(``tileops.backend.register_kernel_type``), beside the in-tree CUDA ``IsnanFwdKernel``/
``AlwaysFalseFwdKernel`` pair (``tileops.kernels.elementwise.comparison``). No existing
file is edited.

**Interface choice is load-bearing, not cosmetic**: ``IsnanFwdOp``
(``tileops/ops/elementwise/comparison.py:86-96``) declares
``interfaces = {ELEMENTWISE: UnaryPredicateFwdInterface}``, a sibling class (neither
extends the other) of the generic ``UnaryElementwiseFwdInterface`` most unary ops in
this project use, with an identical abstract ``forward(self, input)`` shape -- a wrong
choice here would not be caught by any signature check, only by ``op_base.py``'s
``issubclass`` filter, which would silently leave this kernel's key "unassigned" and
raise ``ValueError: IsnanFwdOp keys implement none of its kernel interfaces`` at
``IsnanFwdOp()`` construction time, for every test in the repo, on every device, the
instant this module is imported anywhere in the process (the exact failure mode
``ReciprocalFwdOp``'s round 1 schedule already flagged, now confirmed a third time for a
third, distinct interface). Hence ``AscendIsnanKernel(Kernel, UnaryPredicateFwdInterface)``
below, not ``UnaryElementwiseFwdInterface``.

**This is the project's first op with an asymmetric in/out dtype**: every elementwise op
planned so far has ``output.dtype == input.dtype``. Here the output is always
``torch.bool`` (1 byte/element) regardless of the input dtype (2 or 4 bytes/element for
the three float dtypes this kernel actually compiles a TileLang program for). The
manifest's ``forall`` dtype union
(``manifest/spec/elementwise_unary_math.yaml:424-439``) has 9 members: ``bool``,
``uint8``, ``int8``, ``int16``, ``int32``, ``int64``, ``float16``, ``bfloat16``,
``float32``. Only the 3 float dtypes can represent NaN, so only they get a real kernel;
the other 6 are answered directly in ``forward()`` with
``torch.zeros(input.shape, dtype=torch.bool, device=input.device)`` -- bit-identical to
the CUDA ``AlwaysFalseFwdKernel.handler`` (``kernels/elementwise/comparison.py``), and
deliberately the *same* call, not a re-derived one: there is nothing to tile, pipeline,
or tune for a constant-output op. ``refusal()`` therefore accepts the full 9-dtype
domain, not just the 3 floats -- narrowing it to floats would leave an ``npu`` call with
an integer/bool dtype with *zero* eligible kernels (this kernel refuses on dtype,
``IsnanFwdKernel``/``AlwaysFalseFwdKernel`` both refuse on device, since neither
overrides ``Kernel.devices`` off its CUDA-only default), failing
``test_predicate_non_float_constant`` and
``test_single_tensor_op_completes_every_declared_dtype`` on an ``npu`` run -- the exact
``NegFwdOp``/``AbsFwdOp``/``SignFwdOp`` int-dtype-sibling gap this project's
``dev_process.md`` §7.5 already catalogs, confirmed by direct source read to apply here
too (round_1_schedule.md §3.2).

**Compute body, resolved by 3 real compile probes before this file was written (no NPU
touched -- round_1_schedule.md §4)**: ``T.isnan(x)`` with no cast compiles for
``float16``/``float32`` but fails specifically for ``bfloat16``
(``InternalError: Data type bfloat16 not supported for isnan op``) -- a frontend
``is_float()`` type-check gap in ``tirx::isnan`` (``bfloat16`` is a distinct
``DataTypeCode::kBFloat``, not caught by the function's float/not-float dispatch), not
an Ascend backend/compiler dispatch gap like the tanh/exp/sqrt/rsqrt family's issue.
``T.isnan(T.cast(x, "float32"))`` -- the existing CUDA ``IsnanFwdKernel.op_func``'s exact
pattern -- works for all 3 dtypes and was verified to generate BYTE-IDENTICAL code to the
no-cast version for ``float16``/``float32``: zero tradeoff, so it is used unconditionally
here rather than special-cased per dtype. A second probe confirmed the generated body is
a plain scalar ``x != x`` compare, not an ``AscendMath``-dispatched vendor intrinsic --
none of the promote/intrinsic/cast tax risk this project has catalogued for
Tanh/Exp/Sqrt/Rsqrt applies to this op. A third probe confirmed the asymmetric buffer
(1-byte bool output vs 2/4-byte float input) compiles cleanly at 4 elements/thread: the
output store is a scalar ``int8_t`` write per lane, not a vectorized pack, so
``BitwiseNotFwdOp``'s "sub-32-bit dtype can't vectorize multi-element store" constraint
does not apply here (that fired for a narrow *primary tile* buffer; here the narrow
dtype is only the output, touched as scalar per-lane stores plus one bulk byte-DMA).

Tiling: persistent multi-block grid sized off this SKU's live vector-core count
(``get_num_vector_cores``, floored to the nearest power of two, falling back to 64/72 as
every sibling Ascend elementwise kernel in this directory already does) for any
``n_total`` that divides ``num_blocks * tile`` evenly -- covers all 5 benchmarked (op,
dtype) cases plus the 1,048,576-element correctness smoke fixture. A generic single-block
fallback (mirroring ``relu.py``'s ``_make_relu_generic_single_block``) handles the two
small, non-benchmarked shapes (4096 and 64 elements) that do not divide evenly.

``float32`` ships the schedule's original default, ``tile=8192``/``num_stages=2`` (UB:
80 KiB/block, already comfortably under 80% of baseline on the first M4 run, untouched
since). ``float16``/``bfloat16`` ship ``tile=16384``/``num_stages=2`` -- the schedule's own
risk 2 anticipated exactly this round-1 finding ("if fp16/bf16 specifically miss while
fp32 passes... the established first lever is tile widening to 16384 for fp16/bf16 only")
and it played out: at the schedule's uniform ``tile=8192`` default, fp16/bf16 missed their
80% target on *both* 16M and 256M while fp32 passed outright; widening fp16/bf16 to
``tile=16384`` (UB: 96 KiB/block, still comfortably under ``BitwiseNotFwdOp``'s 256 KiB
overflow point) closed 256M fully and narrowed 16M's gap substantially. A follow-up
``num_stages`` 2-vs-3 A/B at ``tile=16384`` (risk 3) found 3 measurably *worse* than 2 on
every fp16/bf16 case, so round 1 ships 2 for every dtype -- see this op's
``round_1_result.md`` for the full measured numbers and the residual 16M miss this round
left unresolved.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a
CUDA-only process) is a no-op: the Ascend-specific imports are guarded by one
``try/except ImportError`` (``_ASCEND_IMPORT_ERROR``), re-raised lazily only when
``AscendIsnanKernel.__init__`` is actually reached, mirroring ``reciprocal.py`` and every
sibling Ascend kernel file in this directory.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports -- the
same deliberate omission every sibling file in this directory documents (several ops'
workers write into this same directory concurrently; a shared aggregator file would be
contended, ``docs/ascend950/dev_process.md`` §1).
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    _ASCEND_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only off-Ascend
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    get_num_vector_cores = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendIsnanKernel"]

# The 3 float dtypes that can represent NaN: the only ones a real TileLang program is
# built for. Defined locally (plain tuple of torch.dtype) rather than imported from
# kernels/elementwise/math_unary.py, keeping this file's only cross-package dependency
# the tilelang.ascend try/except block (round_1_schedule.md §3.1).
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The other 6 manifest dtypes (round_1_schedule.md §1's corrected count: INT_DTYPES'
# 5 members plus torch.bool). None of them can represent NaN, so forward() answers them
# directly with torch.zeros(..., dtype=torch.bool) -- no TileLang program at all.
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
NON_FLOAT_DTYPES = INT_DTYPES + (torch.bool,)

# Tile size and stage count for the persistent, bandwidth-bound path (round_1_schedule.md
# §6.1). round 1's M4 run found fp16/bf16 missing their 80% target on both 16M and 256M
# while float32 passed comfortably (beats baseline outright) at the uniform tile=8192 the
# schedule shipped as its default -- the exact TanhFwdOp/CosFwdOp/SinFwdOp "two sizes
# miss together, one dtype is fine" symptom shape the schedule's own risk 2 anticipated,
# with risk 2's own pre-vetted first lever: widen tile to 16384 for fp16/bf16 only (UB
# headroom confirmed in the schedule's §5 footprint math: 96 KiB/block at tile=16384,
# still comfortably under BitwiseNotFwdOp's 256 KiB overflow point). float32 keeps
# tile=8192 (already passing, no reason to touch it).
_TILE_FP32 = 8192
_TILE_FP16_BF16 = 16384
_NUM_STAGES_FP32 = 2
_NUM_STAGES_FP16_BF16 = 2
# Fallback vector-core count when the live query below is unavailable. Confirmed 72 on
# this SKU (device Ascend950DT_9592) by multiple independent prior ops in this project
# (docs/ascend950/dev_process.md §7.5).
_FALLBACK_VECTOR_CORES = 72
# T.SimtVF's hard 2048-thread cap (dev_process.md §7.5), confirmed project-wide.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the pipelined, large-``n`` path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``), floored to
    the nearest power of two, rather than hardcoding the figure outright -- cheap, one
    ``npu-guard`` reservation, and a defensive check in case this session's hardware
    differs. Every prior Ascend elementwise kernel with this op's one-load/one-store-per-
    element cost shape (``ReluFwdOp``, ``SignFwdOp``, ``ReciprocalFwdOp``, ...)
    independently lands on 64 here (``floor_pow2(72) == 64``).

    Deliberately **not** evaluated at module import time -- merely importing this module
    must stay a no-NPU-touch no-op. ``lru_cache`` makes it a one-time query per process
    either way. Falls back to ``_FALLBACK_VECTOR_CORES`` if the live query raises or
    returns a falsy value.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _tile_for_dtype(dtype: str) -> int:
    """The persistent-path tile width for ``dtype`` (see ``_TILE_FP32``/``_TILE_FP16_BF16``'s
    comment for why this is dtype-dependent, found during this round's own M4 run)."""
    return _TILE_FP32 if dtype == "float32" else _TILE_FP16_BF16


def _num_stages_for_dtype(dtype: str) -> int:
    """Pipeline stage count for ``dtype``. Both buckets land on 2: float32 at the
    schedule's original tile=8192 (already passing, untouched), and fp16/bf16 at the
    widened tile=16384 -- a direct 2-vs-3 A/B this round's own M4 run ran once tile
    widening alone still left the 16M case short of the 80% target (risk 3's "num_stages
    2-vs-3 sweep if round 1 misses by a small margin") found num_stages=3 measurably
    *worse* than 2 at tile=16384 on every one of fp16/bf16's 16M and 256M cases (see
    round_1_result.md for the numbers) -- the same "more stages does not always help"
    finding ``ReluFwdOp``/``FloorFwdOp`` round 2 already made elsewhere in this project,
    reconfirmed here rather than assumed.
    """
    return _NUM_STAGES_FP32 if dtype == "float32" else _NUM_STAGES_FP16_BF16


if T is not None:

    @T.macro
    def _isnan_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = isnan(x_ub[:n])`` as bool, compiler-vectorized over ``threads``.

        ``T.isnan(T.cast(x, "float32"))`` -- matching ``IsnanFwdKernel.op_func`` exactly
        -- unconditionally for all 3 float dtypes: mandatory for ``bfloat16`` correctness
        (a bare ``T.isnan(x_ub[i])`` fails to compile for bfloat16 -- a frontend
        ``is_float()`` type-check gap, module docstring), and verified byte-identical
        generated code for ``float16``/``float32`` either way (round_1_schedule.md §4) --
        so there is no asymmetric-cost tradeoff to special-case around.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable`` under TileLang's eager builder.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.isnan(T.cast(x_ub[i], "float32"))

    @functools.lru_cache(maxsize=64)
    def _make_isnan_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (round_1_schedule.md §6.1).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_isnan_kernel``) only reaches this builder for a shape that does. Serves
        all 5 benchmarked manifest cases and the 1,048,576-element correctness smoke
        fixture.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_isnan_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), "bool")
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _isnan_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_isnan_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not
        evenly divide (round_1_schedule.md §6.2): one block, a main loop over full
        ``tile``-sized chunks plus one smaller, statically-sized tail chunk. Mirrors
        ``relu.py``'s ``_make_relu_generic_single_block`` verbatim in shape, with
        ``y_ub``/``Y`` declared ``"bool"`` and the body swapped for ``_isnan_body``.
        Handles the two small, non-benchmarked correctness shapes this op's test surface
        needs (4096 and 64 elements, neither reached by any benchmarked manifest
        workload); not performance-tuned.
        """
        full_tiles, remainder = divmod(n_total, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), "bool")
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _isnan_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), "bool")
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _isnan_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_isnan_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements of ``dtype``: persistent grid when it
        divides ``num_blocks * tile`` exactly (round_1_schedule.md §6.3), else the
        generic single-block fallback.
        """
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")
        tile = _tile_for_dtype(dtype)
        num_stages = _num_stages_for_dtype(dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_isnan_persistent(n_total, dtype, num_blocks, tile, num_stages)
        return _make_isnan_generic_single_block(n_total, dtype, tile)


class AscendIsnanKernel(Kernel, UnaryPredicateFwdInterface):
    """IsNaN on Ascend NPUs: ``y = isnan(x)``, a new contiguous ``torch.bool`` tensor.

    Implements ``UnaryPredicateFwdInterface`` (**not** ``UnaryElementwiseFwdInterface`` --
    see this module's docstring for why that distinction is load-bearing for
    ``IsnanFwdOp``'s own ``interfaces`` override). ``devices`` is NPU only;
    ``supported_archs`` is left ``None`` -- unlike the CUDA ``UnaryKernel``/
    ``FloatUnaryKernel`` family this class does not inherit, there is no Ascend SoC-
    version gate here to replace the CUDA SM-version one (``Kernel._check_arch`` skips
    the check entirely when ``supported_archs is None``).

    Accepts the op's full 9-dtype manifest domain, not just the 3 floats this kernel
    compiles a TileLang program for (module docstring, round_1_schedule.md §3.2):
    ``refusal()`` returns ``None`` unconditionally, and ``forward()`` branches
    internally -- the 3 float dtypes (``float16``, ``bfloat16``, ``float32``) run the
    compiled TileLang program below; the other 6 (``bool`` plus the 5 ``INT_DTYPES``,
    none of which can represent NaN) are answered directly with
    ``torch.zeros(input.shape, dtype=torch.bool, device=input.device)``, bit-identical to
    the CUDA ``AlwaysFalseFwdKernel.handler`` and built with no TileLang compile at all.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        # Accepts the FULL manifest domain (9 dtypes), not just the 3 floats this kernel
        # actually compiles a program for -- see module docstring / round_1_schedule.md
        # §3.2's NegFwdOp/AbsFwdOp/SignFwdOp finding: a narrow refusal() here would leave
        # npu+int/bool calls with NO available kernel at all (this kernel refuses, and
        # AlwaysFalseFwdKernel/IsnanFwdKernel are cuda-only and refuse on device too),
        # breaking test_predicate_non_float_constant and
        # test_single_tensor_op_completes_every_declared_dtype on npu.
        return None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendIsnanKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = (
            _make_isnan_kernel(n_total, _dtype_str(dtype)) if dtype in FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isnan(input)`` elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements on an NPU device, contiguous in whatever
                shape the caller passed, in ``self.dtype``.

        Returns:
            A new contiguous ``torch.bool`` tensor on the same device, shaped as
            ``input``. For the 6 non-float manifest dtypes (none of which can represent
            NaN), this is always all-``False`` -- the constant-output path, no TileLang
            kernel involved (module docstring).
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if self.dtype not in FLOAT_DTYPES:
            return torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("IsnanFwdOp", "ascend", AscendIsnanKernel)
except ImportError:
    pass
