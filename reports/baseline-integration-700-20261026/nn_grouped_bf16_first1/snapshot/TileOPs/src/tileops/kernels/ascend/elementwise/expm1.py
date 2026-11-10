"""Ascend NPU expm1 kernel: ``y = exp(x) - 1``, via the ``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``Expm1FwdOp``; see
``docs/ascend950/dev/Expm1FwdOp/round_1_schedule.md`` for the full tiling/schedule design this
file implements. Registers an additional implementation of ``Expm1FwdOp``'s
``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
``Expm1FwdKernel`` (``tileops.kernels.elementwise.math_unary.Expm1FwdKernel``). No existing file
is edited to do this: ``tileops.backend.register_kernel_type`` joins this class to every
``Expm1FwdOp`` instance constructed after this module is imported (``dev_process.md`` §1).

Math: ``Expm1FwdKernel.op_func`` is ``T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")`` --
the literal, naive ``exp(x) - 1`` in fp32, not a Taylor-series/polynomial formula engineered to
avoid cancellation near ``x=0``. This is deliberate, not a missed opportunity: schedule doc §1.2
confirms there is no ``expm1`` intrinsic anywhere in TileLang (unlike ``log1p``, which is a real
registered op), so a native dispatch was never on the table for either backend. Schedule doc §1.3
verifies numerically that the near-zero cancellation this formula leaves on the table is not
load-bearing for this op's own tests: `test_expm1_edge`'s ``x=1e-7`` row (float32 only) has a
measured naive-vs-``torch.expm1`` absolute error of 1.92e-8, about 520x under the fp32
tolerance table's ``atol=rtol=1e-5``. This file therefore ports ``ExpFwdOp``'s already
hardware-validated ``AscendExpKernel`` (``exp.py``) near-verbatim, promoting to fp32 before
calling the hardware ``exp`` intrinsic and casting back to the storage dtype at the shared store
path's boundary (``code-style.md``: "Promote overflow-prone ... ``exp`` ... to fp32; cast back to
storage dtype at the boundary"), with one extra fp32 subtract fused into the same expression.

Tiling design (schedule doc §2) reuses every one of ``exp.py``'s already-converged,
on-device-confirmed constants verbatim -- ``Expm1FwdOp`` and ``ExpFwdOp`` share the exact same
manifest shapes, dtype set, and per-element UB footprint shape (one input tile + one output tile,
no extra scratch; the one extra fp32 subtract operates on a value already live in a register, so
it needs no additional buffer). None of these are re-derived here:

- ``num_blocks``: ``_calibrated_num_blocks()`` pattern verbatim (``get_num_vector_cores``,
  floored to a power of two, fallback 64) -- already the measured optimum on this SKU
  (``Ascend950DT_9592``, 72 vector cores) across four independent pilots (``GeluFwdOp``,
  ``SiluFwdOp``, ``ReluFwdOp`` round 2, ``ExpFwdOp`` round 1, the last one at this exact body
  shape).
- Tile size: ``_UB_BUDGET_BYTES = 216 * 1024``, same sizing formula -> 16384 elements for
  fp16/bf16, 8192 for float32, at ``num_stages=3`` (see below) -- identical arithmetic to
  ``exp.py``'s.
- Threads: ``min(tile_elems, 2048)`` for both dtype classes, the hard Ascend SIMT ceiling
  confirmed in ``dev_process.md`` §7.5.

``num_stages=3`` ships directly, not re-swept (schedule doc §2.1): ``ExpFwdOp`` round 1 already
built and measured both ``num_stages in {2, 3}`` under the official ``measure_tileops_ascend.py``
protocol for a body one subtract lighter than this one, found neither choice flips any case's
pass/fail verdict, and shipped 3 as the better aggregate. Re-running that exact 2-configuration
sweep for a body this close is expected to reproduce the same answer (dev_process.md §7.5's
exhausted-lever-cluster meta-lesson), so round 1 ships the answer directly. ``ExpFwdOp`` round 2's
unpipelined/wider-tile alternative is also not re-attempted here: it measured a clean 25-41%
regression on this exact body shape (``round_2_result.md``), a property of the tiling structure
(losing ``T.Pipelined``'s cross-iteration DMA/compute overlap costs more than the fill/drain it
removes), not of the one extra subtract this op adds.

Does not inherit ``FloatUnaryKernel``/``UnaryKernel`` (``tileops/kernels/elementwise/_base.py``):
both hardcode CUDA-only behavior (``devices={"cuda"}``, ``supported_archs`` read through
``tileops.utils.get_sm_version``, and ``_require_cuda``'s ``tensor.is_cuda`` check) that would
reject every NPU tensor outright.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, every op's worker self-registers from its own file and
does *not* build a shared aggregator under ``kernels/ascend/``, since several ops' workers write
into this same directory concurrently and a shared ``__init__.py`` would be a file they all
contend for. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420)
namespace packages.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
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

__all__ = ["AscendExpm1Kernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendExpKernel (exp.py), which
# in turn reused it from AscendSiluKernel (silu.py), which discovered it via a real on-device
# compile failure on float32's dynamic-shared-memory limit -- this op has the identical
# per-element UB footprint shape (one input tile + one output tile, no extra scratch), so the
# same budget applies unchanged.
_UB_BUDGET_BYTES = 216 * 1024
# Shipped directly, not re-swept (schedule doc §2.1): ExpFwdOp round 1 already built and measured
# both num_stages in {2, 3} under the official measure_tileops_ascend.py protocol for a body one
# subtract lighter than this one and shipped 3 as the better aggregate; re-sweeping for this body
# is expected to reproduce the same answer (dev_process.md §7.5).
_NUM_STAGES = 3
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (ReluFwdOp/GeluFwdOp/SiluFwdOp round 2/
# ExpFwdOp round 1), not an arbitrary guess, so a failed query still degrades to the right answer
# for this device.
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
    ``_calibrated_num_blocks`` (``exp.py``) exactly -- four independent pilots (``GeluFwdOp``,
    ``SiluFwdOp``, ``ReluFwdOp`` round 2, ``ExpFwdOp`` round 1) already swept this under the
    official (no-cache-flush) M4 protocol and converged on 64 as both the live-queried floor
    *and* the measured optimum on this SKU (72 real vector cores, ``Ascend950DT_9592``), with
    both 72-way and 128-way oversubscription measurably worse. This op's body is nearly
    identical to ``ExpFwdOp``'s, so round 1 expects the same answer rather than re-sweeping it
    from scratch (schedule doc §2).

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
    (itemsize 4) -- identical arithmetic to ``exp.py``'s, since the one extra fp32 subtract
    operates on a value already live in a register and needs no additional buffer.
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _expm1_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = exp(x_ub[:n]) - 1``, computed in fp32, compiler-vectorized over
        ``threads``.

        Mirrors ``Expm1FwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        ``T.exp(T.cast(x, "float32")) - T.cast(1.0, "float32")``, cast back to the storage dtype
        at the store (code-style.md's fp32-promotion rule for overflow-prone transcendentals).
        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use ``@T.macro``,
        never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's ``for i in
        T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(
                    T.exp(T.cast(x_ub[i], "float32")) - T.cast(1.0, "float32"),
                    y_ub.dtype,
                )

    @functools.lru_cache(maxsize=64)
    def _build_single_block(n_total: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements.

        Correctness-only fallback: the single shape that reaches this path is
        ``tests/ops/test_unary_math.py::MathEdgeFixture`` (``n_total=4096``, float32 only, no
        latency target -- schedule doc §2). No manifest/benchmark workload is this small.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _expm1_body(x_ub, y_ub, n_total, threads=min(n_total, _MAX_THREADS))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_expm1_kernel``) only reaches this builder for a shape that does (schedule doc
        §2.2: every manifest case and M3 fixture this op is ever asked to run either divides
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
                    _expm1_body(x_ub, y_ub, tile_elems, threads=min(tile_elems, _MAX_THREADS))
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_expm1_kernel(n_total: int, dtype_str: str):
        """Pick the tiling for ``n_total`` elements: persistent tiled path, or single-block
        fallback.

        Per schedule doc §2.2, every shape this kernel will actually be asked to run (5
        manifest cases + the 2 M3 ``MathFixture`` shapes) is an exact multiple of
        ``num_blocks * tile_elems`` once ``n_total`` exceeds one tile; the only shape smaller
        than or not dividing a tile (the 4096-element ``MathEdgeFixture``) routes to the
        trivial single-block fallback. A defensive ``NotImplementedError`` covers a
        hypothetical future shape fitting neither bucket (a la ``AscendExpKernel``'s
        ``_build_exp_kernel``) -- round 1 does not build or tune a real tail path nothing in
        this op's test/benchmark surface reaches.
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
            f"AscendExpm1Kernel: n_total={n_total} dtype={dtype_str} fits neither the tiled "
            f"path (needs a multiple of num_blocks*tile_elems={block_span}) nor the "
            f"single-block fallback (needs n_total<={tile_elems}); no manifest/test shape "
            "should reach this"
        )


class AscendExpm1Kernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU expm1: ``y = exp(x) - 1``, computed in fp32, mirroring
    ``Expm1FwdKernel.op_func``.

    See ``docs/ascend950/dev/Expm1FwdOp/round_1_schedule.md`` for the tiling design. ``devices``
    is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``UnaryKernel``/``FloatUnaryKernel`` family this class does not inherit, there is no Ascend
    SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch`` skips the
    check entirely when ``supported_archs is None``).

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5. This is the literal, naive ``exp(x) - 1``
    formula, not a cancellation-avoiding polynomial near ``x=0`` -- there is no
    ``T.expm1``/``tirx.expm1`` intrinsic on any TileLang target (schedule doc §1.2), and the
    naive formula's worst measured error against ``torch.expm1`` at this op's own edge case
    (``x=1e-7``, float32) is ~520x under the fp32 tolerance table's ``atol=rtol=1e-5`` (schedule
    doc §1.3), so no special-case handling near zero is needed to pass this op's tests.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendExpm1Kernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_expm1_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply ``exp(x) - 1`` elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed.

        Returns:
            A new contiguous tensor on the same device, shaped and typed as ``input``.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("Expm1FwdOp", "ascend", AscendExpm1Kernel)
except ImportError:
    pass
