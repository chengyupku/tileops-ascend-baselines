"""Ascend NPU Hardtanh kernel: ``y = clamp(x, min_val, max_val)``, NaN-propagating, via the
``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``HardtanhFwdOp``; see
``docs/ascend950/dev/HardtanhFwdOp/round_1_schedule.md`` for the full design this file implements.
Registers an additional implementation of ``HardtanhFwdOp``'s ``BoundedUnaryFwdInterface`` under
the ``"ascend"`` dispatch key (``tileops.backend.register_kernel_type``), beside the in-tree CUDA
``HardtanhFwdKernel`` (``tileops.kernels.elementwise.activations``). No existing file is edited.

**Interface choice is load-bearing, not cosmetic**: ``HardtanhFwdOp``
(``ops/elementwise/activations.py``) declares ``interfaces = {ELEMENTWISE:
BoundedUnaryFwdInterface}`` (``BoundsCall`` carries ``min_val``/``max_val``, which the generic
``UnaryElementwiseFwdInterface``'s request type does not) -- confirmed directly against the op's
own ``interfaces`` mapping rather than assumed, per ``dev_process.md`` §7.5's
``ReciprocalFwdInterface``/``IsnanFwdOp`` precedent for this exact pitfall. Hence
``AscendHardtanhKernel(Kernel, BoundedUnaryFwdInterface)`` below, not
``UnaryElementwiseFwdInterface``.

**``min_val``/``max_val`` are both construction-time Python floats, not tensor inputs or runtime
scalars** (schedule doc §1): ``HardtanhFwdOp.__init__`` stores them at construction, and
``_call_spec`` forwards them unchanged into each call's ``BoundsCall``. The in-tree CUDA
``HardtanhFwdKernel`` bakes both into its body as plain Python-float closures, one specialization
per distinct pair -- the same ``ScalarParamUnaryKernel`` pattern ``EluFwdOp``'s ``alpha``/
``LeakyReluFwdOp``'s ``negative_slope`` already use on this backend. **Both scalars are real
``functools.lru_cache`` key dimensions here, not one**: this op's test/benchmark surface genuinely
exercises two distinct ``(min_val, max_val)`` pairs -- ``(-1.0, 1.0)`` (every M4 case,
``test_hardtanh``, both inplace-conformance tests) and ``(-0.5, 0.5)`` (``test_elementwise_compile
.py``'s parametric ``torch.compile(fullgraph=True)`` smoke case only). Every builder below takes
both as explicit, hashable ``functools.lru_cache`` parameters so the two pairs never collide on
the same compiled artifact (schedule doc §3.6). A third pair appears in
``test_scalar_param_rejects_unrepresentable`` (``HardtanhFwdOp(min_val=-1e6)``/``HardtanhFwdOp(
max_val=1e6)``) but is expected to be rejected by the generic, spec-driven
``representable(min_val, T)``/``representable(max_val, T)`` shape-rule guard at the Op layer
(``ops/_signature_codegen.py``) before dispatch ever reaches this kernel -- nothing here checks
representability itself (schedule doc §2.3).

**NaN propagation is a real, tested M3 requirement, and the first on this backend for a
``T.min``/``T.max``-shaped body** (schedule doc §2.2): ``torch.clamp``/``torch.nn.functional.
hardtanh`` propagate NaN (``clamp(NaN, lo, hi) == NaN``), but plain IEEE ``min``/``max`` do not --
they return the non-NaN operand. The in-tree CUDA kernel works around this with a two-operand
``nan_min``/``nan_max`` helper (``kernels/elementwise/_nan.py``) that routes fp16/bf16 through
``tl.max_nan``/``tl.min_nan`` -- TileLang builtins that lower to CUDA's ``__hmax_nan``/
``__hmin_nan`` with no confirmed Ascend registration anywhere in this project. That two-operand
helper is deliberately **not** reused here: Hardtanh's two bounds are the kernel author's own
finite Python-float constants (never NaN, since the unrepresentable-value pairs are rejected
before construction reaches this kernel, per the paragraph above), so the only possible NaN
source is the input ``x`` itself -- a **single-operand** "keep NaN through a monotone
transform" shape, matching ``_erf.py``'s ``keep_nan(x, fn) = Select(isnan(x), x, fn(x))``, which
``AscendErfFwdKernel`` already reuses unmodified on this backend. ``_hardtanh_body`` below builds
the same shape from primitives already independently confirmed working on Ascend: ``T.min``/
``T.max`` (``gelu_and_mul.py``'s boundary clipping), ``T.isnan`` (``isnan.py``, called on an
fp32-cast value so it also works for ``bfloat16`` -- a bare ``T.isnan(bf16_value)`` fails to
compile, a frontend type-check gap documented in ``isnan.py``), and ``T.if_then_else``
(``relu.py``). This NaN handling was **measured, not assumed**, this round (``opus48_result.md``):
a bare ``T.min(T.max(x, lo), hi)`` with no ``isnan`` guard was run on NaN inputs on hardware and
dropped every NaN (``out_nan=0`` vs ``ref_nan=94``), confirming the vector min/max follow IEEE
min/max and the ``isnan`` select is required; Ascend also has no NaN-propagating min/max
intrinsic (``tl.min_nan``/``tl.max_nan`` fail to lower -- ``Unresolved call``), so the explicit
select is the only correct option. The ``hardtanh-nan-input`` M3 test passes on this body.
(The earlier text cited a ``round_1_result.md`` that was never written -- schedule doc §1.)

**The tiled main loop now computes the clamp with explicit SIMD** (``_hardtanh_body_vf``:
``T.SimdVF`` + ``vld``/``vcvt``/``vcmps``/``vsel``/``vsts``; fp32 ordered compare + select),
while ``_build_single_block`` and ``_build_general`` keep the SIMT body above. The split is
deliberate: only the tiled path always works a whole ``tile_elems`` (16384/8192, both
multiples of the 64-lane fp32 vector), and round 1 measured the SIMT body vec-bound with
``vec≈85-92%``, ``aiv_mte2≈60%`` (``opus48_result.md`` §5.2), so the body -- not the tiling --
was the open lever. A standalone ``T.SimdVF`` prototype was verified on device before this
change (dsh worker request 002, 19/19: fp16/bf16/fp32 lane patterns bitwise exact, NaN kept by
the **ordered** compares with no ``isnan`` select, ``lo == hi``/single-sided/narrow bounds,
+-Inf clamped to the finite bounds, and the real (16,256,56,56) 56-block x 14-step shape).
The compare operator is then chosen per bound from its zero sign (``_tie_ops``) so the body
reproduces the reference's IEEE ``min(max(x, lo), hi)`` signed-zero tie-break, measured with an
8-pair x 3-dtype oracle on device (dsh request 015): a strict compare kept ``-0.0`` where the
reference gives ``+0.0`` at zero bounds, and an unconditional ``le``/``ge`` form failed 9/24
oracle items. ``min_val``/``max_val`` sign bits are part of every builder's cache key, because
Python float keys treat ``0.0`` and ``-0.0`` as equal while the reference result differs
between ``(0.0, 0.0)`` and ``(0.0, -0.0)``. The generated 16-bit loop is ``vlds_unpack`` ->
``vcvt<float>`` -> ``vcmps{_le|_lt}`` -> ``vsel`` -> ``vcmps{_ge|_gt}`` -> ``vsel`` ->
``vcvt<half>`` -> ``vsts_pack_b32`` (fp32 uses ``vld``/``vsts_norm`` and 128 iterations per
8192-element tile). The default saturating narrow cannot change semantics here: every value
reaching it is inside the (dtype-representable) bounds or NaN, and a direct ``vcvt(sat=True)``
diagnostic showed NaN -> NaN and Inf -> Inf, never a finite maximum.

Tiling design (schedule doc §3) reuses already-converged, on-device-confirmed constants from
``exp.py``/``elu.py`` for the UB budget and vector-core-count query: identical per-element UB
footprint (one input tile + one output tile, no extra scratch -- ``min_val``/``max_val`` are
trace-time scalars, not buffers), so the same ``_UB_BUDGET_BYTES = 216 * 1024`` budget, the same
``_calibrated_num_blocks()`` (``get_num_vector_cores``, floored to a power of two, fallback 64 --
six independent pilots have already converged on 64 as both the live-queried floor and the
measured optimum on this SKU, ``Ascend950DT_9592``, 72 real vector cores), and the same
``threads = min(tile_elems, 2048)`` apply unchanged.

**``bounded-conv-feat`` (16,256,56,56) = 12,845,056 = 2^18 * 7^2 elements does not divide evenly
at the usual ``num_blocks=64``** (12,845,056 / (64*16384) = 12.25) -- the second non-power-of-two
production shape found in this project (after ``HardswishFwdOp``'s ``mbv3-stage3``,
``dev_process.md`` §7.5's "Batch 6" note) and the first in this op's own lineage. That note flags
a specific silent-regression trap: picking ``num_blocks`` as "the largest factor of ``n_total //
tile_elems`` that is <= the real core count" collapses to ``num_blocks=1`` -- compiling and
passing M3 cleanly, but serializing M4 onto one core -- whenever that quotient happens to be
prime. **Checked directly for this shape, not assumed**: 12,845,056 // 16,384 = 784 = 2^4 * 7^2,
which is composite, so the formula is not structurally unsafe here. The result is still pinned as
an explicit, shape-specific constant below (``_BOUNDED_CONV_FEAT_NUM_BLOCKS = 56``) rather than
computed by a generic divisor search at call time, matching ``ReluFwdOp``'s own precedent for a
hardcoded, shape-specific ``_DECODE_NUM_BLOCKS`` -- this keeps the one non-power-of-two tiling
decision in this file auditable by inspection instead of re-derived (and re-risked) by general
code that could hit the Batch 6 trap on some future shape. 56 is the largest divisor of 784 that
is <= 72 (this SKU's real vector-core count): ``56 * 16384 * 14 == 12,845,056`` exactly (14 clean
``T.Pipelined`` iterations, 78% core utilization), reusing the *same* ``_build_tiled`` persistent
builder as ``bounded-hidden`` with only ``num_blocks`` changed -- zero new boundary/tail code.
56 even-division was measured faster than the 64- or 72-core clipped-tail alternatives this round
(``opus48_result.md`` §core-sweep: conv 56-blk 23.8us vs 64-blk+tail 32.7us vs 72-blk+tail 29.2us
under msprof_detail -- the pipeline drain plus an unpipelined tail costs more than the extra cores
return), so no tail path is used for this shape. (Supersedes the missing ``round_1_result.md``.)

``num_stages=2`` is the measured winner (``opus48_result.md`` §tiling-sweep). This op is
**vector-pipe bound**, not memory bound: msprof_detail shows the shipped body at ``vec≈85-92%``,
``aiv_mte2≈60%``, and the official (no-flush) M4 equals the flush-every-repeat msprof number
(15.23us vs 15.5us for ``bounded-hidden``), so this kernel does not itself benefit from the
16M-cache measurement effect. The full tiling sweep (msprof_detail, both shapes) found
``num_stages=2`` beats 3, single-buffer in-place, and larger tiles; ``ExpFwdOp`` round 2's
"removing ``T.Pipelined`` is a 25-37% regression" was taken as given and that direction not
re-tried. (The earlier text cited a ``round_1_result.md`` §2 that was never written -- schedule
doc §1; the comparison now lives in ``opus48_result.md``.)

Does not inherit ``FloatUnaryKernel``/``ScalarParamUnaryKernel``
(``tileops/kernels/elementwise/_base.py``): both hardcode CUDA-only behavior (``devices=
{"cuda"}``, ``supported_archs`` read through ``tileops.utils.get_sm_version``, and
``_require_cuda``'s ``tensor.is_cuda`` check) that would reject every NPU tensor outright.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, every op's worker self-registers from its own file and does
*not* build a shared aggregator under ``kernels/ascend/``, since several ops' workers write into
this same directory concurrently and a shared ``__init__.py`` would be a file they all contend
for. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420) namespace
packages, same as every sibling file here.

``inplace`` needs no handling here: ``HardtanhFwdOp`` inherits ``_UnaryActivationMixin``
(``ops/elementwise/_base.py``), which copies this kernel's freshly-written output back into the
input tensor at the op layer when ``inplace=True`` -- the kernel itself always returns a new
buffer, same as every other activation kernel in this family.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports are guarded by one ``try/except ImportError``
(``_ASCEND_IMPORT_ERROR``), re-raised lazily only when ``AscendHardtanhKernel.__init__`` is
actually reached, mirroring every sibling Ascend kernel file in this directory.
"""

import functools
import math
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import BoundedUnaryFwdInterface, BoundsCall
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

__all__ = ["AscendHardtanhKernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendExpKernel/AscendEluKernel:
# this op has the identical per-element UB footprint shape (one input tile + one output tile, no
# extra scratch -- min_val/max_val are trace-time scalars, not buffers).
_UB_BUDGET_BYTES = 216 * 1024
# num_stages=2 is the measured optimum (opus48_result.md §tiling-sweep): under msprof_detail at
# the two production shapes, 2 beat 3 (hidden 15.19us vs 16.18us; conv 23.76us vs 25.19us), beat
# single-buffer, and beat larger tiles -- matching ExpFwdOp round 2's "s3 < s2" finding. The
# older in-file claim cited a round_1_result.md that never existed (schedule doc §1); this value
# is now backed by the real sweep logged for this round.
_NUM_STAGES = 2
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (Relu/Gelu/Silu/Exp/Elu round 1-2 pilots),
# not an arbitrary guess, so a failed query still degrades to the right answer for this device.
_FALLBACK_VECTOR_CORES = 64
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048
# bounded-conv-feat (16,256,56,56) = 12,845,056 = 2^18 * 7^2 elements, this op's only manifest
# shape that does not divide evenly at num_blocks=64 (module docstring). 784 = 12,845,056 //
# 16,384 is composite (2^4 * 7^2, checked directly -- not assumed safe, dev_process.md §7.5's
# Batch 6 "quotient might be prime and silently collapse to num_blocks=1" trap); 56 is the
# largest divisor of 784 that is <= 72 (this SKU's real vector-core count), pinned as an explicit
# constant rather than computed generically (module docstring).
_BOUNDED_CONV_FEAT_NUM_BLOCKS = 56
# Fixed scheduling selected by the round-3 shared-runner profile_short re-rank of the same nine
# (num_blocks, tile_elems, num_stages) candidates per case.  The short profile (1 warmup + 3 raw
# independent profiler samples) is a search score, never a formal M4 number; the two previously
# tried candidates were rejected by formal M4.  Keyed by ``(n_total, dtype_str)``; bounds are not
# part of the key and every other shape keeps the calibrated / conv-feat / general routing.
_TUNED_TILING = {
    (2048 * 4096, "float16"): (64, 16384, 3),         # bounded-hidden-float16
    (2048 * 4096, "bfloat16"): (64, 16384, 3),        # bounded-hidden-bfloat16
    (16 * 256 * 56 * 56, "float16"): (56, 16384, 3),   # bounded-conv-feat-float16
    (16 * 256 * 56 * 56, "bfloat16"): (56, 16384, 3),  # bounded-conv-feat-bfloat16
}


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _zero_sign_key(value: float) -> int:
    """``+1``/``-1`` sign of ``value``, an extra component of every builder's cache key.

    ``functools.lru_cache`` -- like Python float equality -- treats ``0.0`` and ``-0.0`` as the
    same key, but this op's reference result depends on a zero bound's sign: measured on device,
    ``F.hardtanh(-0.0, -0.0, 0.0)`` is ``-0.0`` while the ``(0.0, 0.0)`` pair gives ``+0.0``
    (dsh request 015, all 8 signed-zero bound pairs x 3 dtypes).  The builders below take both
    sign bits as key components and restore them with ``math.copysign``, so the two bound pairs
    never share one compiled artifact.
    """
    return -1 if math.copysign(1.0, value) < 0 else 1


def _tie_ops(min_val: float, max_val: float) -> "tuple[str, str]":
    """Ordered-compare ops that reproduce the reference clamp's signed-zero tie-break.

    The reference is IEEE ``min(max(x, lo), hi)``: measured on device (dsh request 015, all 8
    signed-zero bound pairs x 3 dtypes) ``max(+-0, -+0) = +0`` and ``min(+-0, -+0) = -0``.
    A strict ``lt`` keeps ``x`` on a tie, which is wrong exactly when the bound must win:

    * lower bound ``+0.0`` must win a tie with ``-0.0`` -> ``le``, every other lower bound ``lt``;
    * upper bound ``-0.0`` must win a tie with ``+0.0`` -> ``ge``, every other upper bound ``gt``.

    For non-zero bounds the two forms differ only on an exact tie, where both operands are the
    same value, so the original strict compares stay.
    """
    lo_op = "le" if (min_val == 0.0 and math.copysign(1.0, min_val) > 0.0) else "lt"
    hi_op = "ge" if (max_val == 0.0 and math.copysign(1.0, max_val) < 0.0) else "gt"
    return lo_op, hi_op


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
    floors it to the nearest power of two, matching ``AscendEluKernel``'s
    ``_calibrated_num_blocks`` (``elu.py``) exactly -- six independent pilots (``GeluFwdOp``,
    ``SiluFwdOp``, ``ReluFwdOp`` round 2, ``ExpFwdOp``, ``EluFwdOp``) already swept this under the
    official (no-cache-flush) M4 protocol and converged on 64 as both the live-queried floor
    *and* the measured optimum on this SKU (72 real vector cores, ``Ascend950DT_9592``), with both
    72-way and 128-way oversubscription measurably worse. This op's body is lighter than most of
    those precedents', so round 1 expects the same answer rather than re-sweeping it from scratch
    (schedule doc §3.2).

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
    (itemsize 4) at both ``num_stages`` candidates this round considers -- schedule doc §3.3's
    table shows the result is invariant to that choice here, same as ``ExpFwdOp``/``EluFwdOp``.
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _hardtanh_body(
        x_ub, y_ub, n: int, threads: int, min_val: float, max_val: float
    ) -> None:
        """``y_ub[:n] = clamp(x_ub[:n], min_val, max_val)``, NaN-propagating.

        Computed in fp32 and cast back at the store (code-style.md's promotion rule),
        mirroring ``HardtanhFwdKernel._make_op_func`` (``kernels/elementwise/activations.py``)
        except for the NaN handling: the CUDA reference reaches for a two-operand
        ``nan_min``/``nan_max`` helper that is not reused here (module docstring) because its
        fp16/bf16 branch depends on a CUDA-only intrinsic with no confirmed Ascend lowering.
        ``min_val``/``max_val`` are themselves always finite Python-float constants (the
        unrepresentable-value pairs are rejected at the Op layer before a kernel is ever built
        for them), so the single-operand ``isnan(x) ? x : clamp(x)`` shape -- the same idiom
        ``_erf.py``'s ``keep_nan`` already uses on this backend -- is all this op needs: no
        bound can itself introduce a NaN this body has to propagate through a second operand.

        ``T.isnan`` is called on the already-fp32-cast ``wide``, not the raw ``x_ub[i]``,
        matching ``isnan.py``'s own finding: a bare ``T.isnan`` on a ``bfloat16`` value fails to
        compile (a frontend ``is_float()`` type-check gap), while casting to fp32 first works
        for all three dtypes and was independently confirmed byte-identical for fp16/fp32
        either way.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use ``@T.macro``,
        never nested ``prim_func``") -- an undecorated helper's ``for i in T.Parallel(n):``
        fails with ``TypeError: 'ForFrame' object is not iterable`` under TileLang's eager
        builder.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                lo = T.cast(min_val, "float32")
                hi = T.cast(max_val, "float32")
                wide = T.cast(x_ub[i], "float32")
                clamped = T.min(T.max(wide, lo), hi)
                y_ub[i] = T.Cast(y_ub.dtype, T.if_then_else(T.isnan(wide), wide, clamped))

    @T.macro
    def _hardtanh_body_vf(
        x_ub, y_ub, n: int, dtype_str: str, lo_op: str, hi_op: str,
        min_val: float, max_val: float,
    ) -> None:
        """Explicit-SIMD (``T.SimdVF``) clamp body: fp32 ordered compare + select.

        Used only by ``_build_tiled`` (the four benchmark shapes and the large M3 shapes);
        ``_build_single_block`` and ``_build_general`` keep the SIMT body above, because their
        extents are not guaranteed to be a multiple of one 64-lane fp32 vector and this round
        deliberately did not mix a new tail/mask scheme into the same change (schedule
        ``dsh_round1_schedule.md`` items 2 and 5).

        The body is the ordered-select clamp ``a = x < lo ? lo : x; y = a > hi ? hi : a`` in
        fp32, with **no explicit ``isnan``**: IEEE ordered comparisons are false for NaN, so a
        NaN input fails both compares and falls through both selects unchanged.  That the
        hardware ``asc_lt_scalar``/``asc_gt_scalar`` path really is ordered was measured, not
        assumed: the isolated lower-compare kernel kept every NaN (``nan_preserved=410``) and
        the nine three-dtype lane-pattern comparisons were bitwise exact against
        ``torch.clamp`` (dsh worker request 002, ``vf_a_correctness_summary.json`` 19/19).

        The comparison *operator* is chosen per bound from its zero sign (``_tie_ops``): the
        reference is IEEE ``min(max(x, lo), hi)`` -- on-device oracle, dsh request 015, 8
        signed-zero bound pairs x 3 dtypes -- so a tie must take ``lo`` only when ``lo`` is
        ``+0.0`` and must take ``hi`` only when ``hi`` is ``-0.0``.  The strict body kept
        ``-0.0`` where the reference gives ``+0.0`` at ``min_val == max_val == 0.0``
        (95325/953250 zeros), and an unconditional ``le``/``ge`` tie-to-bound variant still
        failed 9/24 oracle items (e.g. ``(-0.0, 0.0)`` with ``x = -0.0``); the conditional
        ``le``/``lt``, ``ge``/``gt`` form matched all 24 + 9 numeric gates and the generated C
        shows the expected ``vcmps_le``/``vcmps_ge``/``vcmps_lt``/``vcmps_gt`` per bound sign
        (dsh request 017).  For non-zero bounds this is byte-for-byte the round-1-verified
        behaviour, because the forms differ only on an exact tie.

        Lane mapping, also measured rather than inferred from the API names: ``vld(...,
        dist="UNPK_B16")`` reads 64 storage elements into the 128-lane 16-bit register and
        ``vcvt(..., "float32", part=0)`` widens the even half to 64 fp32 lanes; the store
        narrows with the default ``ROUND_R``/``sat=True`` and ``dist="PK_B32"`` packs the 64
        results back.  A per-vector ascending pattern with sentinels on lanes 0/1/62/63 (n=2048)
        matched bitwise for fp16/bf16/fp32, and the real (16,256,56,56) conv-feat shape (56
        blocks x 14 pipeline steps) matched bitwise with random data plus NaN/+-Inf/+-0
        sentinels.  Multi-step and multi-pattern coverage beyond that is reported by the
        round's edge probe on the same final source, not claimed here in advance.

        ``sat=True`` is safe here and is **not** what handles +-Inf: every value that reaches
        the narrowing conversion is either inside ``[min_val, max_val]`` (both are
        dtype-representable or the Op layer refuses the pair) or NaN.  +-Inf inputs are already
        clamped to the finite bounds by the selects -- measured for all three dtypes
        (``inf_maps_to_finite_bound_*``) -- and a direct ``vcvt(sat=True)`` diagnostic showed
        NaN -> NaN and Inf -> Inf (never a finite maximum), so the conversion cannot silently
        change the op's semantics.

        The generated C for the 16-bit tile is a single 256-iteration vector loop of
        ``vlds_unpack<half>`` -> ``vcvt<float>`` -> ``vcmps_lt`` -> ``vsel`` -> ``vcmps_gt``
        -> ``vsel`` -> ``vcvt<half>`` -> ``vsts_pack_b32``; the fp32 tile is the same loop with
        ``vld``/``vsts_norm`` and 64 fp32 lanes (128 iterations per 8192-element tile).
        """
        with T.SimdVF():
            lo32 = T.cast(min_val, "float32")
            hi32 = T.cast(max_val, "float32")
            lo_v = T.simd.vdup(lo32, "float32")
            hi_v = T.simd.vdup(hi32, "float32")
            for i in range(n // 64):
                if dtype_str == "float32":
                    wide = T.simd.vld(x_ub[i * 64])
                else:
                    wide = T.simd.vcvt(
                        T.simd.vld(x_ub[i * 64], dist="UNPK_B16"), "float32", part=0
                    )
                low = T.simd.vsel(lo_v, wide, T.simd.vcmps(wide, lo32, op=lo_op))
                high = T.simd.vsel(hi_v, low, T.simd.vcmps(low, hi32, op=hi_op))
                if dtype_str == "float32":
                    T.simd.vsts(y_ub[i * 64], high)
                else:
                    T.simd.vsts(y_ub[i * 64], T.simd.vcvt(high, dtype_str), dist="PK_B32")

    @T.macro
    def _hardtanh_body_native16(
        x_ub, y_ub, n: int, dtype_str: str, min_val: float, max_val: float
    ) -> None:
        """Native 128-lane 16-bit clamp for the default bounds ``(-1, +1)`` only.

        No fp32 storage or conversion: ``vld`` loads 128 half/bf16 lanes, two ordered scalar
        compares with same-dtype ``vdup`` bounds drive native ``vsel``, and ``vsts`` writes 128
        lanes back.  ``-1`` and ``+1`` are exactly representable in both 16-bit dtypes, so no
        arithmetic approximation is introduced.  The branch is selected at build time from the
        real bounds and dtype, never from input values, and is used only by ``_build_tiled``;
        every other bounds pair, fp32, single-block and general keep ``_hardtanh_body_vf`` (or
        the SIMT body).  Ordered compares are false for NaN, so NaN still falls through both
        selects.
        """
        with T.SimdVF():
            lo_s = T.cast(min_val, dtype_str)
            hi_s = T.cast(max_val, dtype_str)
            lo_v = T.simd.vdup(lo_s, dtype_str)
            hi_v = T.simd.vdup(hi_s, dtype_str)
            for i in range(n // 128):
                x = T.simd.vld(x_ub[i * 128])
                low = T.simd.vsel(lo_v, x, T.simd.vcmps(x, lo_s, op="lt"))
                high = T.simd.vsel(hi_v, low, T.simd.vcmps(low, hi_s, op="gt"))
                T.simd.vsts(y_ub[i * 128], high, dist="NORM_B16")

    @functools.lru_cache(maxsize=64)
    def _build_single_block(
        n_total: int, dtype_str: str, min_val: float, max_val: float,
        lo_sign: int, hi_sign: int,
    ):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``n_total`` elements.

        Correctness-only fallback: the only shapes that reach this path are the inplace-
        conformance tests' ``n_total=64`` and the NaN test's ``n_total=768`` (``tests/ops/
        test_elementwise_unary_activation_conformance.py`` / ``test_special_elementwise.py`` --
        schedule doc §3.1/§3.7.3). No manifest/benchmark workload is this small.
        """
        min_val = math.copysign(min_val, lo_sign)
        max_val = math.copysign(max_val, hi_sign)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _hardtanh_body(
                    x_ub,
                    y_ub,
                    n_total,
                    threads=min(n_total, _MAX_THREADS),
                    min_val=min_val,
                    max_val=max_val,
                )
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int,
        dtype_str: str,
        num_blocks: int,
        tile_elems: int,
        num_stages: int,
        min_val: float,
        max_val: float,
        lo_sign: int,
        hi_sign: int,
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_hardtanh_kernel``) only reaches this builder for a shape that does -- either
        the calibrated core count (``bounded-hidden``, M3's main fixture, the compile-smoke
        case) or the shape-specific ``_BOUNDED_CONV_FEAT_NUM_BLOCKS`` (``bounded-conv-feat``,
        module docstring). Same builder function in both cases, only ``num_blocks`` differs
        (schedule doc §3.7) -- no separate code path for the non-power-of-two shape.

        Runs ``_hardtanh_body_vf`` (explicit ``T.SimdVF``) rather than the SIMT body: this is
        the only path whose extent is a whole ``tile_elems`` (16384 or 8192, both multiples of
        the 64-lane fp32 vector), so it is the one place the VF body needs no new tail handling.
        Tiling, staging and the surrounding ``T.copy`` pipeline are unchanged, so the body is
        the only difference from the shipped round-1 structure.
        """
        min_val = math.copysign(min_val, lo_sign)
        max_val = math.copysign(max_val, hi_sign)
        lo_op, hi_op = _tie_ops(min_val, max_val)
        block_span = num_blocks * tile_elems
        assert n_total % block_span == 0, (
            f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        assert tile_elems % 64 == 0, (
            f"_hardtanh_body_vf works one 64-lane fp32 vector at a time; tile_elems "
            f"({tile_elems}) must be a multiple of 64"
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
                    if (dtype_str != "float32" and min_val == -1.0
                            and max_val == 1.0):
                        _hardtanh_body_native16(
                            x_ub, y_ub, tile_elems, dtype_str, min_val, max_val
                        )
                    else:
                        _hardtanh_body_vf(
                            x_ub,
                            y_ub,
                            tile_elems,
                            dtype_str,
                            lo_op,
                            hi_op,
                            min_val=min_val,
                            max_val=max_val,
                        )
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_general(
        n_total: int,
        dtype_str: str,
        num_blocks: int,
        tile_elems: int,
        num_stages: int,
        min_val: float,
        max_val: float,
        lo_sign: int,
        hi_sign: int,
    ):
        """Correctness fallback for any ``n_total`` that no fast path divides evenly.

        ``n_total`` is a build-time constant (one compiled specialization per shape), so the
        full-tile count and the remainder size are both compile-time constants -- no dynamic
        extents are needed. Each of ``num_blocks`` persistent blocks grid-strides over the
        fixed-size full tiles (a runtime ``idx < full_tiles`` guard lets the last partial wave
        of blocks sit idle), then block 0 handles the single constant-size remainder tile. This
        is the bounded-UB main-loop + static-tail general path ``round_1_schedule.md`` §1/§4.5
        calls for, replacing the previous ``NotImplementedError`` -- the manifest declares
        arbitrary Shape, and the in-tree CUDA kernel serves it, so this backend must too.

        Not the performance path: none of the four benchmark shapes reach it (they divide
        evenly at ``num_blocks`` or ``_BOUNDED_CONV_FEAT_NUM_BLOCKS`` and keep ``_build_tiled``).
        Its only job is a correct result for an arbitrary production Shape; the
        ``opus48_result.md`` edge-regression run verified fp16/bf16/fp32 against ``torch.clamp``
        (NaN/±Inf/±0, boundary neighbours, ``lo==hi``) for tile±1, 28685, 65537, 100003,
        1_000_000 and 3_000_007. A tail-based scheme measured slower than even division on the
        benchmark shapes (``opus48_result.md`` §core-sweep), which is exactly why it is confined
        to shapes that have no even-division option at all.
        """
        min_val = math.copysign(min_val, lo_sign)
        max_val = math.copysign(max_val, hi_sign)
        full_tiles = n_total // tile_elems
        rem = n_total - full_tiles * tile_elems
        iterations = (full_tiles + num_blocks - 1) // num_blocks  # static ceil

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    idx = step * num_blocks + bx
                    if idx < full_tiles:
                        begin = idx * tile_elems
                        T.copy(X[begin : begin + tile_elems], x_ub)
                        _hardtanh_body(
                            x_ub,
                            y_ub,
                            tile_elems,
                            threads=min(tile_elems, _MAX_THREADS),
                            min_val=min_val,
                            max_val=max_val,
                        )
                        T.copy(y_ub, Y[begin : begin + tile_elems])
                if rem > 0:
                    if bx == 0:
                        begin = full_tiles * tile_elems
                        T.copy(X[begin : begin + rem], x_ub[:rem])
                        _hardtanh_body(
                            x_ub,
                            y_ub,
                            rem,
                            threads=min(rem, _MAX_THREADS),
                            min_val=min_val,
                            max_val=max_val,
                        )
                        T.copy(y_ub[:rem], Y[begin : begin + rem])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_hardtanh_kernel(n_total: int, dtype_str: str, min_val: float, max_val: float):
        """Pick the tiling for ``n_total`` elements: persistent tiled path (calibrated or
        ``bounded-conv-feat``-specific block count), or single-block fallback.

        Routing (the four benchmark shapes keep their measured-fastest even-division path, so
        the ``opus48_result.md`` M4 numbers are unaffected by the general fallback below):
        every benchmark/M3 shape either divides evenly at the calibrated core count
        (``bounded-hidden``, both M3 ``ActivationFixture`` shapes, the parametric-compile smoke
        shape) or is exactly ``bounded-conv-feat``'s shape, which divides evenly at
        ``_BOUNDED_CONV_FEAT_NUM_BLOCKS`` instead; the shapes smaller than one tile (64 and 768
        elements) route to the single-block fallback. **Any other Shape** -- the manifest
        declares arbitrary Shape, and the in-tree CUDA kernel serves it -- now routes to
        ``_build_general`` (a bounded-UB main loop + constant-size remainder tile), not a
        ``NotImplementedError``. ``opus48_result.md`` verified that general path against
        ``torch.clamp`` for a spread of non-even shapes and all three dtypes.

        ``min_val``/``max_val`` sign bits go into every builder's cache key (``_zero_sign_key``):
        the reference clamp distinguishes ``0.0`` from ``-0.0`` bounds, while Python float keys
        do not.

        Tiled 16-bit calls whose build-time bounds are exactly ``(-1.0, +1.0)`` use
        ``_hardtanh_body_native16`` (128 native lanes, no fp32 conversion); every other call
        keeps ``_hardtanh_body_vf`` with its conditional zero-sign compares.  The native body is
        a bounds-specialised build choice, not a data-dependent dispatch.

        The four manifest shapes consult ``_TUNED_TILING`` first.  Round 3 re-ranked the same
        nine scheduling candidates per case with the shared runner's profile_short score
        (1 warmup + 3 raw independent profiler samples; a search score, not a formal M4
        number); ``num_stages=3`` at ``tile_elems=16384`` was the short-profile leader for all
        four cases while the previously tried candidate A/B mappings regressed under formal M4.
        This is a fixed scheduling table: no runtime tuning, no data-dependent dispatch, and
        bounds are still passed through with their sign keys.
        """
        lo_sign = _zero_sign_key(min_val)
        hi_sign = _zero_sign_key(max_val)
        tuned = _TUNED_TILING.get((n_total, dtype_str))
        if tuned is not None:
            blocks, tile, stages = tuned
            if n_total % (blocks * tile) == 0:
                return _build_tiled(
                    n_total, dtype_str, blocks, tile, stages, min_val, max_val,
                    lo_sign, hi_sign,
                )
        tile_elems = _tile_elems_for_dtype(dtype_str)
        if n_total < tile_elems:
            return _build_single_block(n_total, dtype_str, min_val, max_val, lo_sign, hi_sign)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile_elems
        if n_total % block_span == 0:
            return _build_tiled(
                n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES, min_val, max_val,
                lo_sign, hi_sign,
            )
        conv_feat_span = _BOUNDED_CONV_FEAT_NUM_BLOCKS * tile_elems
        if n_total % conv_feat_span == 0:
            return _build_tiled(
                n_total,
                dtype_str,
                _BOUNDED_CONV_FEAT_NUM_BLOCKS,
                tile_elems,
                _NUM_STAGES,
                min_val,
                max_val,
                lo_sign,
                hi_sign,
            )
        if n_total == tile_elems:
            return _build_single_block(n_total, dtype_str, min_val, max_val, lo_sign, hi_sign)
        return _build_general(
            n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES, min_val, max_val,
            lo_sign, hi_sign,
        )


class AscendHardtanhKernel(Kernel, BoundedUnaryFwdInterface):
    """Hardtanh on Ascend NPUs: ``y = clamp(x, min_val, max_val)``, NaN-propagating.

    See ``docs/ascend950/dev/HardtanhFwdOp/round_1_schedule.md`` for the tiling design.
    ``devices`` is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``ScalarParamUnaryKernel``/``FloatUnaryKernel`` family this class does not inherit, there is
    no Ascend SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch``
    skips the check entirely when ``supported_archs is None``).

    Implements ``BoundedUnaryFwdInterface`` (``BoundsCall`` carries ``min_val``/``max_val``, which
    the generic ``UnaryElementwiseFwdInterface``'s request type does not -- confirmed against
    ``HardtanhFwdOp.interfaces`` directly, module docstring).

    A NaN input (or, hypothetically, a NaN bound) propagates to the output, matching
    ``torch.nn.functional.hardtanh``/``torch.clamp`` and not plain IEEE ``min``/``max`` -- see
    the module docstring and ``_hardtanh_body`` for how this is built from Ascend-proven
    primitives without the CUDA reference's two-operand NaN-safe min/max.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 -- ``HardtanhFwdOp`` is not in any of
    ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the default
    ``Exact()``.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: BoundsCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.min_val, call.max_val)

    def __init__(
        self,
        n_total: int,
        dtype: torch.dtype,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendHardtanhKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.kernel = _build_hardtanh_kernel(
            n_total, _dtype_str(dtype), self.min_val, self.max_val
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Clamp ``input`` elementwise into ``[self.min_val, self.max_val]``; nothing is written
        in place.

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

    register_kernel_type("HardtanhFwdOp", "ascend", AscendHardtanhKernel)
except ImportError:
    pass
