"""Ascend NPU log kernel: ``y = log(x)``, via the ``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``LogFwdOp``; see
``docs/ascend950/dev/LogFwdOp/round_1_schedule.md`` for the full tiling/schedule design this
file implements. Registers an additional implementation of ``LogFwdOp``'s
``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
``LogFwdKernel`` (``tileops.kernels.elementwise.math_unary.LogFwdKernel``). No existing file is
edited to do this: ``tileops.backend.register_kernel_type`` joins this class to every
``LogFwdOp`` instance constructed after this module is imported (``dev_process.md`` §1).
``LogFwdOp.kernel_types`` has no integer sibling (``math_unary.py``: ``{"log": LogFwdKernel}``
only) -- the manifest's ``forall`` is float-only, so unlike ``FloorFwdOp``/``CeilFwdOp`` this
class needs no int-dtype identity branch.

Math, and why this file does **not** mirror the CUDA kernel's fast/precise branch
(schedule doc §1): ``LogFwdKernel.op_func`` is not a plain ``T.log(T.cast(x, "float32"))`` like
``ExpFwdKernel``'s -- it calls ``log_for_output_precision(x, T.cast(x, "float32"))``, which
dispatches to ``T.log`` (CUDA ``logf``) for float32 storage but to ``T.__log`` (CUDA's fast-math
``__logf``) for float16/bfloat16 storage. Probed this directly against the Ascend dialect:
``T.__log`` fails at trace time, independent of storage dtype --

```
AttributeError: module 'tilelang.ascend.language' has no attribute '__log'
```

-- because ``tl.__log`` (``src/cuda/op/builtin.cc`` / ``src/cuda/codegen/codegen_cuda.cc``) has
no Ascend-side registration anywhere in ``src/ascend/`` at all (same story as ``tl.__log2``/
``tl.__log10``/``tl.__tan``): a CUDA-dialect-only fast-math convenience with no Ascend port, not
a codegen gap for one bit width. There is therefore no Ascend equivalent to branch to. This file
uses the plain, precise ``T.log`` on the fp32-promoted value uniformly for every dtype, matching
the already-shipped ``Exp``/``Sqrt``/``Cos``/``Sin`` pattern -- a simplification relative to the
CUDA kernel, not a degradation: precise ``logf`` is strictly more accurate than the fast-math
approximation the CUDA fp16/bf16 branch uses. (One caveat the schedule did not anticipate and
this round's M3 run found directly: "more accurate" is not automatically "matches the on-device
reference" at the fp32 subnormal boundary -- see the dedicated paragraph below and
``_log_body_edge``'s subnormal flush, which resolves it for the one path that needs it.)

``T.log`` itself hits the already-established bf16 compiler gap (schedule doc §1, the fifth
independent confirmation after tanh/sqrt/cos/sin): direct ``T.log(x)`` on bf16 fails at TIR
lowering (``InternalError: Unresolved call ir.Op(..., name="tirx.log", ...)``); on float16 it
compiles natively to ``hlog``, but that native path is **not shipped this round** --
``TanhFwdOp`` round 2 measured the structurally identical native-fp16 lever 3.4-8% *slower* than
fp32-promote despite less generated code, and this op's body is no cheaper to make that economics
flip. fp32-promote is therefore used unconditionally for all three dtypes, both because it is the
only option that compiles for bf16 and because it matches the design every sibling in this batch
converged on.

Domain edge cases (``test_log_edge``'s ``[-1.0, 0.0, 1e-38, 1.0]``): **this round's own M3 run
contradicts the schedule's prediction here, with real evidence, not assumption.** The schedule's
pre-check predicted a bit-exact pass, but the actual ``test_log_edge[4096-dtype0]`` run (M3,
unpatched fp32-promote-then-``T.log`` body) failed: 1024/4096 elements (exactly the ``1e-38``
slots) mismatched, reference ``-inf`` vs. kernel ``-87.498230``. Root-caused with a standalone
on-device diagnostic (bypassing TileLang and pytest entirely, plain ``torch``/``torch_npu``):
``torch.log`` run on an **NPU** tensor flushes a subnormal fp32 input to zero before computing
log (``torch.log(npu_tensor_of(1e-38)) == -inf``), while the identical bit pattern run through
``torch.log`` on a **CPU** tensor -- or through this kernel's Bisheng-compiled ``logf``
intrinsic -- computes the mathematically exact subnormal log (``-87.49822998046875``, bit-exact
between CPU and this kernel). Confirmed the threshold is exactly fp32's min-normal boundary, not
a broader "small value" fudge: at ``2e-38`` (just above ``torch.finfo(torch.float32).tiny ==
1.1754943508222875e-38``, i.e. no longer subnormal), NPU and CPU ``torch.log`` agree exactly
(``-86.80508...``), and the bit pattern of the NPU tensor is confirmed identical to the CPU one
at every step (``x_npu.view(torch.int32)`` matches ``x_cpu.view(torch.int32)``) -- this is a
flush-to-zero quirk specific to the ACL ``Log`` operator ``test_log_edge``'s reference calls,
not a storage/transfer corruption of the input itself. This is the same class of hardware-
intrinsic-specific flush-to-zero behavior ``SqrtFwdOp`` round 1 already documented for ``sqrtf``
at this identical ``1e-38`` boundary (that case flushed the opposite direction -- the *kernel's*
``sqrt`` intrinsic flushed to ``0.0`` while ``torch.sqrt`` did not -- and was harmless there only
because ``sqrt`` is continuous near zero, so the absolute deviation was ~1e-19, far under fp32's
``atol=1e-5``; ``log`` has no such luxury, since flushing or not flushing straddles a
discontinuity at ``-inf``).

**Fix, and a costly first attempt at it, both recorded here**: flush the fp32-promoted value to
exactly ``0.0`` whenever its magnitude is below ``torch.finfo(torch.float32).tiny`` (but nonzero
-- exact zero is already ``0.0`` and unaffected), before calling ``T.log``, reproducing the ACL
reference's own hardware convention at this exact boundary rather than diverging from it. The
**first** attempt inlined this flush into the single, shared body macro every builder called --
compiled and passed `test_log_edge`, but a subsequent M4 pass measured it as a severe regression
on every latency-tracked case, not just the edge fixture: the added branch defeated the Ascend
codegen's packed-vector lowering for fp16/bf16 (a ``T.vectorized loop ... lowered as a serial
loop`` warning appeared for every fp16/bf16 build that did not appear before), costing ~85% at
16M and ~45% at 256M, enough to flip both 256M cases from an expected comfortable PASS to a MISS,
and cost float32 a smaller but still real ~25% even without losing vectorization. **Root cause of
*that* regression**: the flush was reachable from, and therefore compiled into, the one macro
every shape shares, including the five M4-measured cases and ``test_log``'s own correctness
fixture -- none of which can ever be subnormal (next paragraph) -- so the branch was pure
unrewarded cost on the hot path. Fixed by giving the subnormal-flush its own macro
(``_log_body_edge``), called only from ``_build_single_block`` (the untiled fallback the 4096-
element ``MathEdgeFixture`` alone reaches); ``_build_tiled`` -- the path every M4 case and
``test_log``'s ``MathFixture`` take -- keeps the original, branch-free ``_log_body``. Re-verified
by a second M3 run (``test_log_edge[4096-dtype0]`` passes) and a second M4 pass (latencies back
in line with the ``ExpFwdOp``/``FloorFwdOp``/``CeilFwdOp`` precedent, see ``round_1_result.md``
§4) after this split. The flush itself is unreachable by every input ``_build_tiled`` is ever
asked to run: ``test_log``'s ``_positive`` generator (``torch.rand(...).clamp(min=0.01) + 0.01``,
domain ``[0.01, 1.02]``) and the manifest/benchmark ``_draw_positive`` domain are both ~36 orders
of magnitude above ``tiny``.

Tiling design (schedule doc §3) reuses ``ExpFwdOp``'s round-1 configuration verbatim, not the
``Cos``/``Sin``/``Sqrt``/``Rsqrt`` family's: this op's baselines are near-identical to ``ExpFwdOp``'s
case-for-case (``baseline_all_best.csv``), and both ``op_func``s share the same branch-free
"cast -> single intrinsic -> cast" body shape.

- ``num_blocks``: ``_calibrated_num_blocks`` (``tilelang.carver.arch.driver.
  get_num_vector_cores(device_id=0)``, floored to a power of two, fallback 64) -- already
  confirmed 64 on this SKU (72 real vector cores) by multiple independent pilots; not re-swept.
- Tile size: ``_UB_BUDGET_BYTES = 216 * 1024`` and ``max_elems = _UB_BUDGET_BYTES //
  (2 * num_stages * itemsize)``, floored to a power of two -- 16384 elements for fp16/bf16, 8192
  for float32, invariant to which ``num_stages`` candidate below is chosen.
- ``num_stages``: round 1's one open parameter -- built and measured as both 2 and 3 under the
  official ``measure_tileops_ascend.py`` protocol (not ``msprof_detail``, which flushes the GM
  cache every repeat and has already once produced a wrong ranking for this exact kind of sweep,
  ``ReluFwdOp`` round 2); see ``round_1_result.md`` §4 for the measured comparison.
  ``_NUM_STAGES`` below is the winner actually shipped.
- Threads: ``min(tile_elems, 2048)``, the hard Ascend SIMT ceiling (``dev_process.md`` §7.5).
- Small-``n`` routing: a single-block, no-pipelining fallback for any shape smaller than one
  tile, reused from ``ExpFwdOp``'s precedent -- the only shape that reaches it is
  ``test_log_edge``'s ``MathEdgeFixture`` (``n_total=4096``, float32 only), smaller than even the
  float32 tile (8192).

Does not inherit ``FloatUnaryKernel``/``UnaryKernel`` (``tileops/kernels/elementwise/_base.py``):
both hardcode CUDA-only behavior (``devices={"cuda"}``, ``supported_archs`` read through
``tileops.utils.get_sm_version``, and ``_require_cuda``'s ``tensor.is_cuda`` check) that would
reject every NPU tensor outright.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, several ops' workers write into this same
``kernels/ascend``/``kernels/ascend/elementwise`` directory concurrently, each registering from
its own file, and a shared ``__init__.py`` would be a file they all contend for.
``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420) namespace packages,
matching every other Ascend elementwise kernel already in this directory.
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

__all__ = ["AscendLogKernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendSiluKernel/AscendExpKernel,
# which discovered it via a real on-device compile failure on float32's dynamic-shared-memory
# limit -- this op has the identical per-element UB footprint shape (one input tile + one output
# tile, no extra scratch), so the same budget applies unchanged.
_UB_BUDGET_BYTES = 216 * 1024
# The one open round-1 parameter (schedule doc §3): built and measured as both 2 and 3 under the
# official measure_tileops_ascend.py protocol; see round_1_result.md §4 for the comparison.
_NUM_STAGES = 3
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (ReluFwdOp/GeluFwdOp/SiluFwdOp round 2), not
# an arbitrary guess, so a failed query still degrades to the right answer for this device.
_FALLBACK_VECTOR_CORES = 64
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048
# torch.finfo(torch.float32).tiny -- the smallest positive *normal* fp32 value. Below this
# magnitude (and above exact zero), fp32 is in its subnormal range. The Ascend ACL `Log` operator
# backing test_log_edge's torch.log reference flushes subnormal fp32 inputs to zero before
# computing log (confirmed by a direct on-device diagnostic, module docstring); this kernel's own
# T.log intrinsic does not, so _log_body_edge below (the single-block path only -- see its
# docstring for why the shared tiled-path body deliberately stays branch-free) flushes to match
# that reference convention exactly at this boundary, not an approximation of it.
_FP32_MIN_NORMAL = 1.1754943508222875e-38


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
    floors it to the nearest power of two, matching ``ExpFwdOp``'s/``ReluFwdOp``'s
    ``_calibrated_num_blocks`` exactly -- already converged on 64 across multiple independent
    pilots on this SKU (72 real vector cores, ``Ascend950DT_9592``); not re-swept here (schedule
    doc §3).

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
    (itemsize 4) at both ``num_stages`` candidates this round considers (schedule doc §3's table
    shows the result is invariant to that choice here).
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _log_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = log(x_ub[:n])``, computed in fp32, compiler-vectorized over ``threads``
        lanes.

        Unlike ``LogFwdKernel.op_func`` (CUDA), there is no fast/precise branch by storage
        dtype here: ``T.__log`` (the fast-math log the CUDA kernel uses for fp16/bf16 storage)
        is not exported on the Ascend dialect at all (module docstring -- an
        ``AttributeError`` at trace time, not a codegen gap). Promotion is also a hard
        compile-time requirement for bf16 (module docstring -- direct ``T.log`` on bf16 fails
        to lower).

        No subnormal flush here -- deliberately (see ``_log_body_edge`` below for the one path
        that needs it). This macro backs only ``_build_tiled``, which every M4-measured case and
        ``test_log``'s own ``MathFixture`` correctness fixture reach (``_positive`` domain,
        ``>= 0.01``, never subnormal) -- so the extra branch has no correctness upside here and a
        real, measured cost: a first M4 pass with the flush inlined into this exact macro showed
        it defeats the Ascend codegen's packed-vector lowering for fp16/bf16 (``T.vectorized loop
        ... lowered as a serial loop`` warning, absent without the branch) and cost 16M-fp16/bf16
        ~85% and 256M-fp16/bf16 ~45% versus the unbranched body, enough to flip both 256M cases
        from an expected PASS to a MISS (round_1_result.md §4). Keeping this macro branch-free and
        moving the flush to the untiled, non-performance-critical single-block path only
        (``_build_single_block`` / ``_log_body_edge``) fixes ``test_log_edge`` with zero cost to
        every case that is actually latency-measured.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's ``for i in
        T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.log(T.cast(x_ub[i], "float32")), y_ub.dtype)

    @T.macro
    def _log_body_edge(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = log(x_ub[:n])``, with a subnormal-fp32 flush-to-zero before the call.

        Used **only** by ``_build_single_block``, the untiled correctness-only path the 4096-
        element ``MathEdgeFixture`` (``test_log_edge``, float32 only) is the sole caller of
        (module docstring, ``_build_log_kernel``) -- never the latency-measured ``_build_tiled``
        path, so this body's extra branch costs nothing anywhere M4 or ``test_log`` looks.

        **Why the flush exists** (module docstring -- a real M3 finding, not a precaution): the
        fp32-promoted value is flushed to exactly ``0.0`` whenever ``0 < |x| < _FP32_MIN_NORMAL``,
        matching the Ascend ACL ``Log`` operator's own hardware behavior at this boundary
        (``test_log_edge``'s reference is ``torch.log`` run on-device, which does this flush;
        this kernel's ``T.log`` intrinsic does not, by default). Without it, ``log(1e-38)``
        returns the mathematically exact ``-87.498...`` instead of the reference's ``-inf``,
        failing ``test_log_edge`` outright (an unbounded deviation, not a tolerance near-miss --
        unlike ``SqrtFwdOp``'s analogous but harmless flush mismatch at the same boundary, module
        docstring). ``T.abs`` on an already-fp32 value is confirmed safe to call directly
        (``floor.py``'s module docstring: fp32's native ``fabsf`` path for ``T.abs`` compiles
        cleanly; the known 16-bit ``T.abs`` compile gap does not apply here, this operand is
        always fp32).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                x32 = T.cast(x_ub[i], "float32")
                flushed = T.if_then_else(
                    T.abs(x32) < T.float32(_FP32_MIN_NORMAL), T.float32(0.0), x32
                )
                y_ub[i] = T.cast(T.log(flushed), y_ub.dtype)

    @functools.lru_cache(maxsize=64)
    def _build_single_block(n_total: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements.

        Correctness-only fallback: the single shape that reaches this path is
        ``tests/ops/test_unary_math.py::MathEdgeFixture`` (``n_total=4096``, float32 only, no
        latency target -- schedule doc §3). No manifest/benchmark workload is this small. Uses
        ``_log_body_edge`` (the subnormal-flushing body), not the plain ``_log_body``, since this
        is the only path ``MathEdgeFixture``'s ``1e-38`` value reaches.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _log_body_edge(x_ub, y_ub, n_total, threads=min(n_total, _MAX_THREADS))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_log_kernel``) only reaches this builder for a shape that does (schedule doc
        §3: every manifest case and M3 fixture this op is ever asked to run either divides
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
                    _log_body(x_ub, y_ub, tile_elems, threads=min(tile_elems, _MAX_THREADS))
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_log_kernel(n_total: int, dtype_str: str):
        """Pick the tiling for ``n_total`` elements: persistent tiled path, or single-block
        fallback.

        Per schedule doc §3, every shape this kernel will actually be asked to run (5 manifest
        cases + the 2 M3 ``MathFixture`` shapes) is an exact multiple of
        ``num_blocks * tile_elems`` once ``n_total`` exceeds one tile; the only shape smaller
        than or not dividing a tile (the 4096-element ``MathEdgeFixture``) routes to the trivial
        single-block fallback. A defensive ``NotImplementedError`` covers a hypothetical future
        shape fitting neither bucket -- round 1 does not build or tune a real tail path nothing
        in this op's test/benchmark surface reaches.
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
            f"AscendLogKernel: n_total={n_total} dtype={dtype_str} fits neither the tiled path "
            f"(needs a multiple of num_blocks*tile_elems={block_span}) nor the single-block "
            f"fallback (needs n_total<={tile_elems}); no manifest/test shape should reach this"
        )


class AscendLogKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU log: ``y = log(x)``, computed in fp32 for every storage dtype.

    See ``docs/ascend950/dev/LogFwdOp/round_1_schedule.md`` for the tiling design. ``devices``
    is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``UnaryKernel``/``FloatUnaryKernel`` family this class does not inherit, there is no Ascend
    SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch`` skips the
    check entirely when ``supported_archs is None``).

    Unlike ``LogFwdKernel`` (CUDA), which routes float16/bfloat16 storage through a fast-math
    log intrinsic (``T.__log``, CUDA ``__logf``) and only float32 through the precise ``T.log``,
    this kernel uses the precise ``T.log`` uniformly for every dtype: ``T.__log`` has no Ascend
    registration at all (module docstring). This is strictly more accurate than the CUDA
    fp16/bf16 path, so it cannot regress accuracy relative to what that path would have given.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 (``LogFwdOp`` is not in any of
    ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the default
    ``Exact()``). Domain edges (``x <= 0``) follow standard IEEE-754 ``logf``/``hlog`` behavior:
    ``log(0) = -inf``, ``log(negative) = nan``. The one domain edge that does *not* follow plain
    IEEE-754 ``logf`` semantics is a subnormal fp32 input (``0 < |x| < torch.finfo(torch.float32
    ).tiny``), exercised only by the untiled, non-performance-critical single-block path
    (``_build_single_block`` / ``_log_body_edge``, module docstring): that path flushes it to
    ``0.0`` before calling ``T.log``, matching the Ascend ACL ``Log`` operator's own hardware
    behavior rather than computing the mathematically exact subnormal log -- required for
    ``test_log_edge`` to match its ``torch.log``-on-NPU reference; confirmed bit-exact against
    that reference on ``test_log_edge``'s full input set at float32 after this fix
    (round_1_result.md §3). The latency-measured ``_build_tiled`` path never sees a subnormal
    input (every M4 case and ``test_log``'s own fixture use the ``_positive``/``_draw_positive``
    domains, ``>= 0.01``) and keeps the branch-free ``_log_body`` unconditionally.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendLogKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_log_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply log elementwise; nothing is written in place.

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

    register_kernel_type("LogFwdOp", "ascend", AscendLogKernel)
except ImportError:
    pass
