"""Ascend Hardswish kernel: ``y = x * clamp(x + 3, 0, 6) / 6``, via the ``tilelang.ascend``
dialect.

Registers an additional implementation of ``HardswishFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``HardswishFwdKernel``
(``tileops.kernels.elementwise.activations.HardswishFwdKernel``). No existing file is edited to
do this: ``tileops.backend.register_kernel_type`` joins this class to every ``HardswishFwdOp``
instance constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Round 1 design: ``docs/ascend950/dev/HardswishFwdOp/round_1_schedule.md``. Two decisions worth
reading this docstring for before reading the code:

**H1 (corrected 2026-10-19 by a hardware edge probe) -- fp16 and bf16 compute in fp32; only fp32
uses the native storage-dtype body.** The body is ``x * clamp(x + 3, 0, 6) / 6``. An earlier
revision kept fp16 in its storage dtype, reasoning the clamp bounds (0, 6) cap the result so
there is no overflow. That reasoning is wrong: the overflow-prone term is ``x * clamped`` -- *x*
(up to the dtype max) times up-to-6 -- not ``clamped`` alone. For any fp16 ``x`` in
``(65504/6, 65504]`` ~= ``(10917, 65504]`` the clamp has already saturated to 6, so the body
evaluates ``x * 6`` in fp16, overflowing to ``+inf``. A hardware edge probe
(``docs/ascend950/dev/resume_20261019/workers/HardswishFwdOp/probe_edge_native.log``, 2026-10-19,
kernel sha256 ``48d33713...``) confirmed this on device: the native-fp16 kernel returned ``+inf``
for 16384, 32768, 60000, 65280 and 65504, where ``torch.nn.functional.hardswish`` on the *same*
NPU tensor returned the correct finite values (it does not overflow) -- the native-fp16 body
disagreed with the authoritative reference at six finite inputs. The in-tree CUDA
``HardswishFwdKernel`` (``kernels/elementwise/activations.py``) already casts its 3/6/0/(1/6)
constants to float32, so its arithmetic runs in fp32 and never hits this; the Ascend kernel now
matches it. ``randn`` M3 inputs (~[-5, 5]) never reach the overflow regime, which is why this was
latent -- the probe, not the project tests, surfaced it.

Fix: ``_hardswish_body_promoted`` (fp32-promote, cast back at the store -- structurally
``sigmoid.py``/``silu.py``/``hardtanh.py``'s already-shipped pattern) serves **fp16 and bf16**;
``_hardswish_body_native`` (native storage dtype) serves **fp32 only**, where the storage dtype
*is* fp32, so it carries no promotion and cannot overflow below the dtype max (at the max itself
the fp32 reference overflows identically, so they still agree).

bf16 has a *second*, independent reason it cannot use the native body, found by a pre-M4 compile
probe (``tilelang.compile(target="ascend")``, no NPU touch): calling ``T.min``/``T.max`` directly
on two ``bfloat16_t`` operands fails to compile -- this backend's CCE header defines ``half
max(half, half)`` and ``float max(float, float)`` overloads but **no direct ``bfloat16_t``
overload**, so overload resolution falls through to ambiguous integer-promotion candidates. A new
entry for ``dev_process.md`` §7.5's bf16-gap list, blocking ``T.min``/``T.max`` specifically, not
``+``/``*``. Promoting to fp32 sidesteps both the overflow and this compile gap at once. See this
op's ``opus48_result.md`` for the verbatim probe tables and the M4 numbers this split produced.

**A2 (round 2, 2026-10-19) -- ``x < -3`` returns a signed ``-0.0`` so ``-inf`` matches the NPU
reference.** The authoritative reference is ``F.hardswish`` on the *input device*
(``workloads/elementwise.py``); on NPU it returns ``-0`` for ``x = -inf``. The plain body
``x*clamp(x+3,0,6)*(1/6)`` gives the right ``-0`` for every *finite* ``x < -3`` (``x*0 = -0`` by
IEEE sign), but for ``x = -inf`` it computes ``-inf * 0 = NaN``. The ordered-select form
``T.if_then_else(x < -3, -0.0, normal)`` has been verified on this device: request 003's
production-Op edge regression passed 23/23 cases including explicit ``torch.signbit`` checks
(``-inf`` and every finite ``x < -3`` give signed ``-0``), and request 002's M3 passed 6/6, both at
hash ``25be1cc6...`` (this was previously described here as unverified; that was only because the
probe had not run yet). Its formal M4 at that hash measured ratios 0.591686 / 0.585341 / 0.657661 /
0.638021, and its generated C keeps the 16-bit body as a scalar four-iteration loop with a
lane-level branch. A select-free variant using ``normal/float(x==x)`` was rejected by the real
device edge probe (NaN -> -0 in all 20 non-empty cases) because the compiler folded ``x == x``;
an ``exp(-|x|)`` poison variant was statically rejected before execution. Those failures are kept
in the round-1 evidence and are **not** used here.

Round 2 therefore switches only the **16-bit tiled main loop** to an explicit ``T.SimdVF`` body
(``_hardswish_body_vf``): 64-lane fp32 load/convert, ordered compares and ``vsel`` for the clamp,
the original ``x * clamp`` then ``* 1/6`` order, and an ordered ``x < -3`` select to ``-0``. NaN
falls through every ordered compare into the original-``x`` product, so it is preserved; +inf and
huge finite values keep their existing overflow behaviour. The static tail, every non-64-lane
extent, and the fp32 main path keep the verified SIMT body from ``25be1cc``. The VF body below is
the same arithmetic the standalone 64-lane prototype validated 3/3 (request 011); the integrated
production hash still requires its own 6 M3 + 23 edge + formal M4 before it is accepted, so this
paragraph must not be read as a performance result.

**Round 3 scheduling map (2026-10-19).** ``_TUNED_SCHEDULE`` fixes only the four
real ``(n_total, dtype)`` manifest cases selected by the real TileLang AutoTuner
short-profile search and its 5-config refinement: both stage2 cases use
``tile=12288, num_stages=4`` (their refinement winners), both stage3 cases keep
``tile=16384, num_stages=3`` (stage3-fp16's control won; stage3-bf16's 12288/s3 was
a 0.03% tie and was not adopted). Every other shape keeps the original
divisor/tile/stages route. The body, math order, NaN/Inf and signed-zero rules are
unchanged; the formal same-hash M3 + boundary + M4 evidence is still required and
short search scores are not acceptance.

**Round 4 main-VF unroll (2026-10-19).** ``_make_hardswish_kernel(..., vf_unroll=U)``
adds one Python-int cache key (default 1). Only the 16-bit full-tile VF main loop uses
``T.serial(n//(64U))`` + ``T.unroll(U, explicit=True)``; SIMT tail, fp32 main, general
shapes and the arithmetic/NaN/signed-zero order are unchanged. U is 1/2/4 and must divide
the tile/64; production selects U per real case through ``_TUNED_UNROLL`` (final 4/4/2/1 after the 12-trial search).

**H3 -- ``num_blocks`` from a divisor of ``n_total`` itself, not of ``n_total // tile``.**
``relu.py``/``sigmoid.py``/``silu.py`` all compute their persistent-grid block count as the
largest divisor of ``num_full_tiles = n_total // tile_elems`` that is ``<= get_num_vector_cores()``
-- safe there because every one of those ops' manifest shapes is a power of two, so
``num_full_tiles`` stays a clean power of two with 64 always available as a divisor. Hardswish's
two manifest shapes are **not** powers of two (``mbv3-stage2``: ``9,633,792 = 2**16 * 147``;
``mbv3-stage3``: ``6,021,120 = 2**13 * 735``), and at a UB-sized tile, ``mbv3-stage3``'s
``num_full_tiles`` lands on 367, which is **prime** -- the old pattern's divisor search would
silently collapse to ``num_blocks=1``, parking the entire workload on one of this SKU's 72 vector
cores. This file instead searches for the largest divisor of ``n_total`` **itself**
(``_num_blocks_for_n_total``), then tiles *within* each block's own share
(``per_block = n_total // num_blocks``) with a static per-block main loop (``T.Pipelined``) plus
one un-pipelined static tail copy for ``per_block % tile``, all inside the same
``T.Kernel(num_blocks)`` body. One compiled program per ``(n_total, dtype)`` handles every shape
this op is asked for this round (both manifest cases, the ``2**20`` M3 case, and the degenerate
64-element inplace-conformance case alike) -- there is no separate decode/tail-dispatch split
like ``sigmoid.py``'s ``_SigmoidRunner`` needs. See ``opus48_result.md`` for the actual
``num_blocks`` the builder chose for each judged case, and whether this single-kernel design
traced/compiled cleanly or the two-kernel fallback (schedule §3.3) was needed instead.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports are guarded by one ``try/except ImportError``
(``_ASCEND_IMPORT_ERROR``), and the one real NPU-touching entry point
(``AscendHardswishKernel.__init__``) re-raises it lazily, mirroring ``relu.py``/``sigmoid.py``.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: ``docs/ascend950/dev_process.md`` §1 explicitly asks every op's worker to self-
register from its own file and *not* build a shared aggregator file under ``kernels/ascend/``,
since several ops' workers write into this same new directory concurrently and a shared
``__init__.py`` would be a file they all contend for. ``kernels/ascend`` and
``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420) namespace packages, same
precedent as every sibling file already here (``relu.py``, ``sigmoid.py``, ``silu.py``, ...).
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

__all__ = ["AscendHardswishKernel"]

# H2 main-loop tile, num_stages (resume_20261019_schedule.md §4). The main-loop double buffer is
# 2 buffers (x_ub, y_ub) x tile x bytes-per-elem x num_stages of UB; the budget is 216 KiB.
#   16-bit (fp16/bf16): tile=16384 -> 2*16384*2*2 = 128 KiB  (schedule §4's "widen first" lever)
#   float32          : tile=8192  -> 2*8192*4*2  = 128 KiB  (16384 would be 256 KiB, over budget)
# Both land at 128 KiB; the plus-one tail buffer pair (remainder-sized) adds <=16 KiB for these
# shapes, so peak UB stays well under 216 KiB. The dtype split is why this is a function, not one
# constant: fp32 cannot take the wide tile. num_stages=2 is the shared double-buffer depth.
_UB_BUDGET_BYTES = 216 * 1024  # documents the headroom the tiles below assume
_TILE_FP32 = 8192
_TILE_16BIT = 16384
_NUM_STAGES = 2

# Round-3 production schedule mapping: only the four real (n_total, dtype) cases
# from the short-profile search. Candidate A = measured top-1 per case (the two
# 16-bit stage cases use stages=3; stage2-fp16 keeps control). All other shapes
# keep the original divisor/tile/stages route below. This is a scheduling map,
# not a new body: the VF main / SIMT tail math is unchanged. Formal M4 is still
# required; short-profile scores are not acceptance.
_TUNED_SCHEDULE = {
    (9633792, "float16"): (64, 12288, 4),
    (9633792, "bfloat16"): (64, 12288, 4),
    (6021120, "float16"): (70, 16384, 3),
    (6021120, "bfloat16"): (70, 16384, 3),
}

# Round-4 explicit main-VF unroll per real case (1 = old code shape). Only the 16-bit
# full-tile VF main loop reads it; tail/fp32/general stay U=1 and the math order is
# unchanged. It is a Python-int builder cache key.
_TUNED_UNROLL = {
    (9633792, "float16"): 4,
    (9633792, "bfloat16"): 4,
    (6021120, "float16"): 2,
    (6021120, "bfloat16"): 1,
}


def _tile_for_dtype(dtype_str: str) -> int:
    """Main-loop tile for ``dtype_str``: wide (16384) for 16-bit, 8192 for float32 (UB budget)."""
    return _TILE_FP32 if dtype_str == "float32" else _TILE_16BIT
# Fallback vector-core count when the live query below is unavailable. Confirmed on this SKU
# (device Ascend950DT_9592) by ReluFwdOp/TanhFwdOp round 1/2 (dev_process.md §7.5):
# get_num_vector_cores reports 72, not 64 or 128.
_FALLBACK_VECTOR_CORES = 72
# Hard Ascend SIMT thread ceiling (dev_process.md §7.5): T.SimtVF(threads=...) must stay <=2048
# independent of tile size; a tile/remainder larger than this is handled by multiple
# elements-per-thread, not more threads.
_MAX_THREADS = 2048


@functools.lru_cache(maxsize=1)
def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of the current NPU device.

    Queries ``tilelang.carver.arch.driver.get_num_vector_cores`` (confirmed to report 72 on this
    SKU), falling back to ``_FALLBACK_VECTOR_CORES`` only if the live query raises or returns a
    falsy value. A wrong or unavailable count only changes how many persistent blocks the kernel
    launches with -- ``_num_blocks_for_n_total`` always returns a true divisor of ``n_total``, so
    every block still does an exact, equal share of the work regardless of this value.
    Deliberately not evaluated at import time: it queries the real device, so this is only called
    lazily, the first time a kernel is actually built (always under an ``npu-guard`` reservation
    already, per ``dev_process.md`` §4). ``lru_cache`` makes it a one-time query per process.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _num_blocks_for_n_total(n_total: int, vector_cores: int) -> int:
    """The largest divisor of ``n_total`` itself that does not exceed ``vector_cores``.

    See this module's docstring, H3: this is deliberately a divisor search over ``n_total``, not
    over ``n_total // tile`` the way ``relu.py``/``sigmoid.py``/``silu.py`` do it -- that pattern
    is only safe when ``n_total // tile`` stays a round number, which it does not for either of
    this op's manifest shapes. Guarantees ``n_total % result == 0``, so every block gets an exact,
    equal ``per_block = n_total // result`` share with no cross-block remainder to handle; 1
    always qualifies, so this never fails to return a valid block count, even for an ``n_total``
    whose only small divisor is 1.
    """
    cap = max(1, min(n_total, vector_cores))
    for candidate in range(cap, 0, -1):
        if n_total % candidate == 0:
            return candidate
    return 1  # unreachable: candidate == 1 always divides n_total


if T is not None:

    @T.macro
    def _hardswish_body_native(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = x_ub[:n] * clamp(x_ub[:n] + 3, 0, 6) / 6``, native storage dtype.

        Mirrors ``HardswishFwdKernel.op_func`` (``kernels/elementwise/activations.py``) exactly,
        except every ``T.cast(..., "float32")`` becomes ``T.cast(..., x_ub.dtype)``. Used for
        **fp32 only** now (H1, corrected 2026-10-19): for fp32 the storage dtype is already fp32,
        so this is identical to the promoted body with no redundant cast and no overflow below the
        dtype max. fp16 must NOT use this body -- ``x * clamped`` evaluates ``x * 6`` in fp16 for
        x in the clamp-saturated range and overflows to ``+inf`` (a hardware probe caught it
        disagreeing with the torch reference at finite inputs); bf16 cannot use it either, because
        ``T.min``/``T.max`` on two ``bfloat16_t`` operands fails to compile. Both use
        ``_hardswish_body_promoted`` below instead. A ``@T.macro``, not a plain Python
        helper (``code-style.md``: "sub-routines use ``@T.macro``, never nested ``prim_func``"):
        TileLang's eager builder only rewrites ``@T.prim_func``/``@T.macro`` bodies into its
        trace-builder calls, so an undecorated helper's ``for i in T.Parallel(n):`` fails with
        ``TypeError: 'ForFrame' object is not iterable`` (``dev_process.md`` §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                dt = x_ub.dtype
                three = T.cast(3.0, dt)
                six = T.cast(6.0, dt)
                zero = T.cast(0.0, dt)
                one_sixth = T.cast(1.0 / 6.0, dt)
                neg_three = T.cast(-3.0, dt)
                neg_zero = T.cast(-0.0, dt)
                clamped = T.min(T.max(x_ub[i] + three, zero), six)
                normal = x_ub[i] * clamped * one_sixth
                # A2 (2026-10-19): x < -3 -> signed -0.0, matching F.hardswish on NPU. The plain
                # product gives -0 for finite x<-3 but NaN for x=-inf (-inf*0); this select fixes
                # -inf while leaving NaN (NaN<-3 is False) and +inf/huge-finite on the normal path.
                y_ub[i] = T.if_then_else(x_ub[i] < neg_three, neg_zero, normal)

    @T.macro
    def _hardswish_body_promoted(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = x_ub[:n] * clamp(x_ub[:n] + 3, 0, 6) / 6``, fp32-promoted.

        fp16 + bf16 path (H1, corrected 2026-10-19). fp16 needs this because its native-dtype
        body overflows: ``x * clamped`` evaluates ``x * 6`` in fp16 once the clamp saturates and
        reaches ``+inf`` for finite x, disagreeing with the torch reference (hardware probe,
        ``opus48_result.md``). bf16 needs it for a second, independent reason, confirmed by a
        direct compile probe: calling ``T.min``/``T.max`` directly on ``bfloat16_t`` operands
        fails to compile on this backend -- the Ascend CCE header defines a direct ``half
        max(half, half)`` overload but no direct ``bfloat16_t`` overload, so overload resolution
        falls through to several equally-plausible integer-promotion candidates and fails
        ambiguously. Promoting to fp32 first sidesteps both at once (``float max(float, float)``
        is an exact, unambiguous overload, and fp32 has the range to hold ``x * 6``) -- the same
        structure as ``sigmoid.py``'s ``_sigmoid_op``/``silu.py``'s ``_silu_op``/``hardtanh.py``'s
        ``_hardtanh_body``, all of which already promote unconditionally. See this module's
        docstring (H1) and ``opus48_result.md`` for the probe evidence behind this split.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                wide = T.cast(x_ub[i], "float32")
                three = T.cast(3.0, "float32")
                six = T.cast(6.0, "float32")
                zero = T.cast(0.0, "float32")
                one_sixth = T.cast(1.0 / 6.0, "float32")
                neg_three = T.cast(-3.0, "float32")
                neg_zero = T.cast(-0.0, "float32")
                clamped = T.min(T.max(wide + three, zero), six)
                normal = wide * clamped * one_sixth
                # A2 (2026-10-19): wide < -3 -> signed -0.0, matching F.hardswish on NPU (-inf
                # would otherwise be -inf*0 = NaN). NaN<-3 is False so NaN propagates; +inf and
                # huge-finite stay on the normal path.
                selected = T.if_then_else(wide < neg_three, neg_zero, normal)
                y_ub[i] = T.cast(selected, y_ub.dtype)

    @T.macro
    def _hardswish_body_vf(x_ub, y_ub, n: int, dtype_str: str, vf_unroll: int = 1) -> None:
        """Explicit-SIMD Hardswish over one full 64-lane-aligned tile, fp32 arithmetic.

        Round 2 main-loop body for 16-bit full tiles only (``n % 64 == 0``). It is the
        same arithmetic as ``_hardswish_body_promoted`` -- ``clamp(x+3,0,6)``, then
        ``(x * clamped) * (1/6)``, then an ordered ``x < -3`` select to ``-0`` -- but
        expressed with the explicit VF load/cast/compare/select/store ops that
        ``hardtanh.py``'s ``_hardtanh_body_vf`` already established on this backend.
        The ordered compares are the NaN contract: a NaN lane fails every compare and
        falls through the clamp unchanged, so the original ``x`` multiplication keeps
        NaN; ``-inf`` fails neither and is selected to signed ``-0``. The static tail
        and the fp32 main path stay on the SIMT body above.
        """
        with T.SimdVF():
            three = T.cast(3.0, "float32")
            six = T.cast(6.0, "float32")
            zero = T.cast(0.0, "float32")
            one_sixth = T.cast(1.0 / 6.0, "float32")
            neg_three = T.cast(-3.0, "float32")
            neg_zero = T.cast(-0.0, "float32")
            zero_v = T.simd.vdup(zero, "float32")
            six_v = T.simd.vdup(six, "float32")
            neg_zero_v = T.simd.vdup(neg_zero, "float32")
            for g in T.serial((n // 64) // vf_unroll):
                for s_unroll in T.unroll(vf_unroll, explicit=True):
                    base = (g * vf_unroll + s_unroll) * 64
                    if dtype_str == "float32":
                        wide = T.simd.vld(x_ub[base])
                    else:
                        wide = T.simd.vcvt(
                            T.simd.vld(x_ub[base], dist="UNPK_B16"), "float32", part=0
                        )
                    xp3 = T.simd.vadds(wide, three)
                    low = T.simd.vsel(zero_v, xp3, T.simd.vcmps(xp3, zero, op="lt"))
                    clamped = T.simd.vsel(six_v, low, T.simd.vcmps(low, six, op="gt"))
                    normal = T.simd.vmuls(T.simd.vmul(wide, clamped), one_sixth)
                    result = T.simd.vsel(
                        neg_zero_v, normal, T.simd.vcmps(wide, neg_three, op="lt")
                    )
                    if dtype_str == "float32":
                        T.simd.vsts(y_ub[base], result)
                    else:
                        T.simd.vsts(
                            y_ub[base], T.simd.vcvt(result, dtype_str), dist="PK_B32"
                        )

    @functools.lru_cache(maxsize=128)
    def _make_hardswish_kernel(
        n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int,
        vf_unroll: int = 1
    ):
        """Compile one program: ``num_blocks`` persistent blocks, each covering its own
        ``per_block = n_total // num_blocks`` share via a ``T.Pipelined`` main loop over
        ``tile``-sized chunks, plus -- when ``per_block`` does not divide ``tile`` evenly -- one
        static, un-pipelined tail copy for the remainder (schedule doc §3.3, H3 above). Every
        shape this op is asked for this round reaches this single builder: both manifest cases,
        the ``2**20`` M3 shape, and the degenerate 64-element inplace-conformance shape, where
        ``num_blocks`` itself equals ``n_total`` and ``per_block == 1`` collapses to a
        tail-only, zero-full-tile body (``full_tiles == 0`` below) -- correct, just not tuned
        for speed, since that shape carries no latency target.

        The tail copy intentionally sits outside the ``T.Pipelined`` region (runs once,
        un-pipelined, after it) -- ``T.Pipelined`` does not support nesting
        (``baseline_techniques.md`` P2), and this avoids the question entirely.
        """
        # H1 (corrected 2026-10-19 by hardware probe): fp16 and bf16 both use the fp32-promoted
        # body; only fp32 uses the native storage-dtype body. fp16-native overflows (x*clamp = x*6
        # in fp16 for x>65504/6 -> +inf, disagreeing with the torch reference at finite inputs);
        # bf16-native additionally fails to compile (T.min/T.max on bfloat16_t). fp32's storage
        # dtype *is* fp32, so its native body carries no promotion and cannot overflow below the
        # dtype max. This dispatch happens once, at Python/trace level, before either macro runs.
        body = _hardswish_body_native if dtype_str == "float32" else _hardswish_body_promoted
        # Round 2: the 16-bit full-tile main loop uses the explicit VF body (the tile is
        # always 64-lane aligned: 16384 here, and the zero-tail candidate's 8192/4096).
        # fp32 main, every static tail and every non-aligned fallback keep the verified
        # SIMT body, so a remainder never needs a VF mask scheme.
        use_vf_main = dtype_str != "float32" and tile % 64 == 0
        if vf_unroll not in (1, 2, 4):
            raise ValueError(f"vf_unroll must be 1/2/4, got {vf_unroll}")
        if use_vf_main and tile % (64 * vf_unroll) != 0:
            raise ValueError(f"tile {tile} not divisible by 64*vf_unroll={64 * vf_unroll}")
        per_block = n_total // num_blocks
        full_tiles, remainder = divmod(per_block, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                base = bx * per_block
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype_str)
                    y_ub = T.alloc_shared((tile,), dtype_str)
                    T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                    for step in T.Pipelined(full_tiles, num_stages=num_stages):
                        begin = base + step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        if use_vf_main:
                            _hardswish_body_vf(x_ub, y_ub, tile, dtype_str, vf_unroll)
                        else:
                            body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype_str)
                    yt_ub = T.alloc_shared((remainder,), dtype_str)
                    begin = base + full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    body(xt_ub, yt_ub, remainder, threads=min(_MAX_THREADS, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_hardswish_kernel(n_total: int, dtype_str: str):
        """Resolve the fixed round-3 mapping for real cases, else the original route."""
        tuned = _TUNED_SCHEDULE.get((n_total, dtype_str))
        if tuned is not None:
            num_blocks, tile, num_stages = tuned
        else:
            num_blocks = _num_blocks_for_n_total(n_total, _num_vector_cores())
            tile = _tile_for_dtype(dtype_str)
            num_stages = _NUM_STAGES
        vf_unroll = _TUNED_UNROLL.get((n_total, dtype_str), 1)
        return _make_hardswish_kernel(
            n_total, dtype_str, num_blocks, tile, num_stages, vf_unroll
        )


class AscendHardswishKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU Hardswish: ``y = x * clamp(x + 3, 0, 6) / 6``; fp16/bf16 compute in fp32, fp32
    natively (H1, corrected 2026-10-19 -- the native fp16 body overflowed at large finite inputs).

    See ``docs/ascend950/dev/HardswishFwdOp/resume_20261019_schedule.md`` for the tiling design and
    this op's own ``opus48_result.md`` for what correctness (M3) and performance (M4) verification
    actually found on hardware. ``devices`` is NPU only; ``supported_archs`` is left ``None`` --
    unlike the CUDA ``UnaryKernel``/``FloatUnaryKernel`` hierarchy this class does not inherit,
    there is no Ascend SoC-version gate here to replace the CUDA SM-version one
    (``Kernel._check_arch`` skips the check entirely when ``supported_archs is None``).

    No bit-exact test is registered for this op (unlike ``SigmoidFwdOp``'s
    ``test_16bit_sigmoid_matches_torch_bit_for_bit``); the generic elementwise tolerance table
    (fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5) applies.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendHardswishKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_hardswish_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Hardswish elementwise; nothing is written in place.

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

    register_kernel_type("HardswishFwdOp", "ascend", AscendHardswishKernel)
except ImportError:
    pass
