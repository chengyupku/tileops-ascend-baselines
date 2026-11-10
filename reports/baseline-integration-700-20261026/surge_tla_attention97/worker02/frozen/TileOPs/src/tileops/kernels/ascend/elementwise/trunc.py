"""Ascend NPU trunc kernel: ``y = trunc(x)`` (truncate toward zero), via ``tilelang.ascend``.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``TruncFwdOp``; see
``docs/ascend950/dev/TruncFwdOp/round_1_schedule.md`` for the full tiling/schedule design this
file implements. Registers an additional implementation of ``TruncFwdOp``'s
``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
``TruncFwdKernel``/``IntIdentityFwdKernel`` (``tileops.kernels.elementwise.math_unary``). No
existing file is edited to do this: ``tileops.backend.register_kernel_type`` joins this class to
every ``TruncFwdOp`` instance constructed after this module is imported (``dev_process.md`` §1).

Math: mirrors ``TruncFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly --
``T.trunc(T.cast(x, "float32"))``, cast back to the storage dtype at the store boundary (code-
style.md's fp32-promotion rule). Schedule doc §1.2 confirmed, via source
(``tilelang/src/ascend/codegen/intrin_rule_ascend.cc``'s ``REGISTER_ASCEND_INTRIN`` table) and a
live compile-only probe, that Ascend has a real native ``trunc`` intrinsic (``truncf``/``htrunc``
depending on bit width), the same ``AscendMath`` dispatch mechanism that already backs
``AscendExpKernel`` et al.: fp32 and fp16-promoted both lower to ``truncf``; fp16 has an
additional, genuinely native, un-promoted ``htrunc`` path that compiles with no pack/unpack;
bfloat16's un-promoted path hard-fails (``Unresolved call ir.Op(...tirx.trunc...)``, the same
project-wide bf16 custom-datatype legalization gap ``tanh``/``sqrt`` already hit). Round 1
promotes **all three** float dtypes to fp32 unconditionally, including float16, despite the native
fp16 path being compile-confirmed and available: schedule doc §1.3 -- ``TanhFwdOp`` round 2 found
the structurally identical "compiles to less code, no promotion" native lever measurably 3.4-8%
*slower* in practice than the fp32-promote path, so "fewer generated instructions" is not evidence
of "faster" on this hardware without a direct M4 measurement. Round 1 tried this stretch goal
directly (budget allowed, and the primary path's ``float16`` case missed its target -- schedule
doc §1.3/§9's explicit trigger condition): a temporary un-promoted ``float16`` body, A/B measured
against the shipped fp32-promote path under the same official M4 protocol, came back **3.6-4.0%
slower** (16M: 0.021996ms vs 0.021143ms; 256M: 0.325529ms vs 0.314352ms) -- the same direction of
surprise as ``TanhFwdOp``, now independently confirmed for ``trunc`` too, and it would not have
closed the 16M miss either way (still >1.7x target). Reverted; this file ships the fp32-promote
path, unchanged, for all three float dtypes. Full numbers in ``round_1_result.md`` §2.

Tiling design (schedule doc §3) reuses ``AscendExpKernel``'s (``exp.py``, already shipped and
measured) constants and three-way dispatch structure verbatim, since ``TruncFwdOp``'s manifest
cases are byte-for-byte identical in shape to ``ExpFwdOp``'s and the compute body (one fp32 cast
in, one single-intrinsic call, one cast back) is the same shape: 216KiB UB budget, ``num_blocks``
queried and floored to a power of two (fallback 64), ``tile_elems`` 16384 (fp16/bf16) / 8192
(fp32), ``num_stages=3`` (not reopened this round -- schedule doc §3.1 judges this op's body shape
close enough to ``ExpFwdOp``'s to adopt its settled value directly rather than re-sweep an
already-exhausted lever), ``threads=min(tile_elems, 2048)``. Per code-style.md and this batch's
concurrent-file convention (schedule doc §3), the small ``_floor_pow2``/``_calibrated_num_blocks``/
``_tile_elems_for_dtype`` helpers are duplicated locally here rather than imported from ``exp.py``,
so sibling ops' same-round worker files never depend on each other's internals or import order.

Mandatory int-dtype design (schedule doc §2): the manifest's dtype domain for ``TruncFwdOp`` is
the full union of 5 integer dtypes plus the 3 float dtypes, but ``IntIdentityFwdKernel`` (which
serves integers on CUDA, shared across floor/ceil/round/trunc) does not override ``devices`` and
stays stuck at ``{"cuda"}`` -- on an ``npu`` tensor it is ``unavailable()`` regardless of dtype,
same as the CUDA-only ``TruncFwdKernel``. If this class's ``refusal()`` narrowed itself to floats
only, zero kernels would remain available for ``npu`` + any integer dtype and dispatch would break
outright (the same bug already hit ``NegFwdOp``/``AbsFwdOp``/``SignFwdOp``, per
``dev_process.md`` §7.5). So ``refusal()`` here explicitly accepts the full union (ints imported
from the existing CUDA kernel module, not redefined locally, to stay single-sourced), and
``forward()`` branches on dtype: integers short-circuit to ``input.clone()`` (no TileLang program
is ever built for them -- mirrors ``IntIdentityFwdKernel.handler = torch.Tensor.clone`` exactly,
including its lack of flatten/reshape), floats run the compiled kernel.

Does not inherit ``FloatUnaryKernel``/``UnaryKernel``/``TorchFallbackKernel``
(``tileops/kernels/elementwise/_base.py``): all hardcode CUDA-only behavior (``devices={"cuda"}``,
``supported_archs`` read through ``tileops.utils.get_sm_version``) that would reject every NPU
tensor outright, or (for ``TorchFallbackKernel``) never build a TileLang program for the float
path this kernel needs.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1 and every existing sibling in this directory (e.g.
``exp.py``), each op's worker self-registers from its own file and does not build a shared
aggregator under ``kernels/ascend/``, since several ops' workers write into this same directory
concurrently and a shared ``__init__.py`` would be a file they all contend for.
``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420) namespace packages.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise._dtype import FLOAT_DTYPES
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.elementwise.math_unary import INT_DTYPES
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

__all__ = ["AscendTruncKernel"]

# The manifest's full declared dtype domain for this op (schedule doc §2.1): the 5 integer
# dtypes `IntIdentityFwdKernel` serves on CUDA, plus the 3 float dtypes this file's compiled
# kernel serves. `refusal()` must accept all of it, or `npu` + any integer dtype loses every
# available kernel (the `IntIdentityFwdKernel`/`NegFwdOp`/`AbsFwdOp` gap, dev_process.md §7.5).
_ALL_DTYPES = INT_DTYPES + FLOAT_DTYPES

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendExpKernel/AscendSiluKernel
# -- this op has the identical per-element UB footprint shape (one input tile + one output tile,
# no extra scratch), so the same budget applies unchanged (schedule doc §3).
_UB_BUDGET_BYTES = 216 * 1024
# ExpFwdOp's converged round-1 value for this exact body shape (one cast in, one intrinsic call,
# one cast out); schedule doc §3.1 judges Trunc's body close enough to adopt directly rather than
# re-sweep an already-exhausted lever.
_NUM_STAGES = 3
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (ReluFwdOp/GeluFwdOp/SiluFwdOp/ExpFwdOp), not
# an arbitrary guess, so a failed query still degrades to the right answer for this device.
_FALLBACK_VECTOR_CORES = 64
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
    floors it to the nearest power of two, matching ``AscendExpKernel``'s
    ``_calibrated_num_blocks`` exactly -- already converged on 64 as both the live-queried floor
    *and* the measured optimum on this SKU (``Ascend950DT_9592``, 72 real vector cores) by
    multiple independent pilots. This op's body is the same shape as ``ExpFwdOp``'s, so round 1
    expects the same answer rather than re-sweeping it from scratch (schedule doc §3).

    Deliberately **not** evaluated at module import time: the query touches the real device, so
    merely importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
    ``lru_cache`` makes it a one-time query per process either way.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return fallback
    return _floor_pow2(int(queried))


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the tiled path's UB use in budget.

    2 buffers (x tile, y tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
    ``_UB_BUDGET_BYTES``. Resolves to 16384 for fp16/bf16 (itemsize 2) and 8192 for float32
    (itemsize 4), matching ``AscendExpKernel``'s table (schedule doc §3).
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _trunc_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = trunc(x_ub[:n])``, computed in fp32, compiler-vectorized over ``threads``.

        Mirrors ``TruncFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        ``T.trunc(T.cast(x, "float32"))``, cast back to the storage dtype at the store (code-
        style.md's fp32-promotion rule). A ``@T.macro``, not a plain Python helper (code-style.md:
        "sub-routines use ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder
        only rewrites ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.trunc(T.cast(x_ub[i], "float32")), y_ub.dtype)

    @functools.lru_cache(maxsize=64)
    def _build_single_block(n_total: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements.

        Correctness-only fallback: no manifest/benchmark workload this op is ever asked to run
        is small enough to need this path (schedule doc §3.2's iteration-count table), written
        defensively to match ``AscendExpKernel``'s shape.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _trunc_body(x_ub, y_ub, n_total, threads=min(n_total, _MAX_THREADS))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_trunc_kernel``) only reaches this builder for a shape that does (schedule doc
        §3.2: every manifest case and M3 fixture this op is ever asked to run either divides
        evenly or is small enough to route to ``_build_single_block``), so no tail handling is
        implemented here.
        """
        block_span = num_blocks * tile_elems
        assert n_total % block_span == 0, (
            f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _trunc_body(x_ub, y_ub, tile_elems, threads=min(tile_elems, _MAX_THREADS))
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_trunc_kernel(n_total: int, dtype_str: str):
        """Pick the tiling for ``n_total`` elements: persistent tiled path, or single-block
        fallback.

        Per schedule doc §3.2, every shape this kernel will actually be asked to run (5
        manifest cases + the ``MathFixture`` shape) is an exact multiple of
        ``num_blocks * tile_elems`` once ``n_total`` exceeds one tile; a defensive
        ``NotImplementedError`` covers a hypothetical future shape fitting neither bucket, a la
        ``AscendExpKernel``'s identically-shaped dispatch -- round 1 does not build or tune a
        real tail path nothing in this op's test/benchmark surface reaches.
        """
        tile_elems = _tile_elems_for_dtype(dtype_str)
        if n_total < tile_elems:
            return _build_single_block(n_total, dtype_str)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile_elems
        if n_total % block_span == 0:
            return _build_tiled(n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES)
        if n_total == tile_elems:
            return _build_single_block(n_total, dtype_str)
        raise NotImplementedError(
            f"AscendTruncKernel: n_total={n_total} dtype={dtype_str} fits neither the tiled path "
            f"(needs a multiple of num_blocks*tile_elems={block_span}) nor the single-block "
            f"fallback (needs n_total<={tile_elems}); no manifest/test shape should reach this"
        )


class AscendTruncKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU trunc: ``y = trunc(x)`` (truncate toward zero), computed in fp32 for every float
    dtype, mirroring ``TruncFwdKernel.op_func``; integer inputs pass through unchanged.

    See ``docs/ascend950/dev/TruncFwdOp/round_1_schedule.md`` for the tiling design. ``devices``
    is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``UnaryKernel``/``FloatUnaryKernel``/``TorchFallbackKernel`` family this class does not
    inherit, there is no Ascend SoC-version gate here to replace the CUDA SM-version one
    (``Kernel._check_arch`` skips the check entirely when ``supported_archs is None``).

    Serves the manifest's full declared dtype domain for this op: the 5 integer dtypes (``uint8``,
    ``int8``, ``int16``, ``int32``, ``int64``) as well as the 3 float dtypes (``float16``,
    ``bfloat16``, ``float32``). Integer inputs are returned as an unchanged clone -- ``trunc`` is a
    no-op on an already-integral value, matching ``IntIdentityFwdKernel.handler`` -- and never
    reach a compiled TileLang program. Float inputs run the compiled kernel below.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 (``TruncFwdOp`` is not in any of
    ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the default
    ``Exact()``). ``T.trunc`` leaves ``±inf``/``NaN`` unchanged (propagated) and ``±0.0`` signed
    unchanged, the same IEEE-754 ``trunc`` contract CUDA's ``truncf`` already has and the existing
    CUDA ``TruncFwdKernel`` already relies on; promoting fp16/bf16 to fp32 is lossless both ways
    here (a finite fp16/bf16 value upcasts to fp32 exactly, and ``trunc`` of any value
    representable in a format is itself exactly representable in that same format), so in practice
    matches against ``torch.trunc`` are expected to land at exactly ``0`` error, not merely within
    tolerance.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        if call.dtype not in _ALL_DTYPES:
            return f"serves dtypes {_ALL_DTYPES}, not {call.dtype}"
        return None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if dtype not in INT_DTYPES and _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendTruncKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = (
            None if dtype in INT_DTYPES else _build_trunc_kernel(n_total, _dtype_str(dtype))
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply trunc elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed (for the float path -- the integer path imposes
                no such constraint, matching ``IntIdentityFwdKernel.handler``).

        Returns:
            A new tensor on the same device, shaped and typed as ``input``. For an integer
            ``self.dtype``, this is ``input.clone()`` exactly (no shape manipulation, no
            TileLang program built or run -- ``torch.trunc`` is a no-op on integers).
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if self.dtype in INT_DTYPES:
            return input.clone()
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

    register_kernel_type("TruncFwdOp", "ascend", AscendTruncKernel)
except ImportError:
    pass
