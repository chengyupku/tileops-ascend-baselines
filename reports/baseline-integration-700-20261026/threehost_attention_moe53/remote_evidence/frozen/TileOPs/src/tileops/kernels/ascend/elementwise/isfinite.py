"""Ascend NPU IsFinite forward kernel: ``y = isfinite(x)``, bool output, via the
``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for
``IsfiniteFwdOp``; see ``docs/ascend950/dev/IsfiniteFwdOp/round_1_schedule.md`` for the
schedule this file implements, and ``round_1_result.md`` for how the two open questions
the schedule flagged (algebraic-identity-vs-cast body, tile-chunking for the bool output)
were actually resolved by live compile probes before this file was written. Registers an
additional implementation of ``IsfiniteFwdOp``'s own ``UnaryPredicateFwdInterface`` under
the ``"ascend"`` dispatch key (``tileops.backend.register_kernel_type``), beside the
in-tree CUDA ``IsfiniteFwdKernel``/``AlwaysTrueFwdKernel`` pair
(``tileops.kernels.elementwise.comparison``). No existing file is edited.

**Interface choice is load-bearing, not cosmetic** (round_1_schedule.md §2): ``comparison.py``
declares ``class IsfiniteFwdKernel(FloatPredicateKernel, UnaryPredicateFwdInterface)`` --
a sibling interface of the generic ``UnaryElementwiseFwdInterface`` most unary ops use,
with an identical ``forward(self, input)`` shape textually but a different docstring
contract ("bool out" vs "same dtype out"). A wrong choice here is not caught by any
signature check, only by ``op_base.py``'s ``issubclass`` filter, raising ``ValueError:
IsfiniteFwdOp keys implement none of its kernel interfaces`` at ``IsfiniteFwdOp()``
construction time on every device the instant this module is imported anywhere in the
process. Hence ``AscendIsfiniteKernel(Kernel, UnaryPredicateFwdInterface)`` below, not
``UnaryElementwiseFwdInterface``.

**Mixed-dtype design (round_1_schedule.md §3)**: the manifest's ``forall`` dtype union has
9 members (``bool``, ``uint8``, ``int8``, ``int16``, ``int32``, ``int64``, ``float16``,
``bfloat16``, ``float32``); only the 3 floats can hold a NaN or an infinity, so only they
get a real TileLang kernel. The other 6 are answered directly in ``forward()`` with
``torch.ones(input.shape, dtype=torch.bool, device=input.device)`` -- bit-identical to the
CUDA ``AlwaysTrueFwdKernel.handler``. **This is the ``True``/``ones`` sibling, not the
``False``/``zeros`` one** ``IsnanFwdOp``/``IsinfFwdOp`` use for their own non-float
fallback -- an int/bool value can never be NaN or +-Inf, so it is *always* finite, the one
place this op's design must diverge from its two closest siblings. ``refusal()`` therefore
accepts the full 9-dtype domain, not just the 3 floats -- narrowing it to floats would
leave an ``npu`` call with an integer/bool dtype with zero eligible kernels (this kernel
refuses on dtype, ``IsfiniteFwdKernel``/``AlwaysTrueFwdKernel`` both refuse on device,
neither overriding ``Kernel.devices`` off its CUDA-only default), failing
``test_predicate_non_float_constant`` and
``test_single_tensor_op_completes_every_declared_dtype`` on an ``npu`` run. Unlike
``NegFwdOp``'s int branch, ``torch.ones(...)`` never reads ``input``'s data at all (only
its shape/device), so there is no ACLNN operator-coverage question to probe for any of the
6 constant-output dtypes.

**Compute body -- resolved by live compile probes, not the schedule's original plan
(round_1_result.md has the full transcripts)**: the schedule's primary candidate was a
hand-rolled algebraic identity ``(x - x) == 0`` (zero promotion, zero named-intrinsic
dependency), with ``T.isfinite(T.cast(x, "float32"))`` -- the existing CUDA kernel's own
``op_func`` -- as the ordered fallback if Bisheng's fast-math folding broke the identity.
**A direct compile probe at this project's real persistent-grid shape (``tile=8192``,
``threads=2048``, 4 elements/thread) found the algebraic-identity body FAILS to compile
for all 3 float dtypes** (``InternalError: SimtVF cannot convert vector type int8x4 to an
Ascend type`` -- confirming this op's own schedule's finding, round_1_schedule.md §5.2,
and the structurally analogous failure ``IsinfFwdOp``'s planner independently found for
its own body). **The cast-based fallback body does NOT hit this wall**: a second probe at
the identical ``tile=8192``/``threads=2048``/``num_blocks=64`` shape, no chunking at all,
compiled clean for all 3 dtypes. Inspecting the generated device source
(``JITKernel.get_kernel_source()``) explains why: the cast-based body's bool store lowers
to a ``#pragma unroll`` loop of 4 independent **scalar** ``int8_t`` stores (one per
unrolled iteration, each its own address), never a packed ``int8x4`` vector store, because
the value being stored comes from a `fabsf`+NaN-self-compare expression tree the
vectorizer does not pattern-match into a batched store. The raw ``(x - x) == 0`` identity,
being a plain subtract-then-compare on a vectorizable linear access pattern, apparently
*does* get pattern-matched into an attempted 4-wide vector store, which then fails because
Ascend has no ``int8x4`` type. **Net effect: the vectorization wall is driven by which
expression generates the bool, not by "any bool store at >1 element/thread fails"** --
this reconciles the apparent ``IsnanFwdOp``-vs-``IsfiniteFwdOp`` planner discrepancy
reported upstream (``IsnanFwdOp``'s ``T.isnan(T.cast(x, "float32"))`` body is the same
intrinsic-call shape and was independently confirmed scalar/non-vectorized by its own
worker). Net: round-1 ships the cast-based body (not the algebraic identity), which also
sidesteps the open fast-math-folding risk the schedule flagged for the algebraic identity
-- an opaque comparison-tree expression (two independent ``x != x`` NaN self-compares plus
an ``fabsf``-vs-``inf`` compare, visible directly in the generated source) is a materially
different optimization target than a bare ``x - x`` subtraction, and is the same pattern
the in-tree CUDA kernel already relies on for correct NaN/Inf handling. ``test_isfinite``
(``tests/ops/test_special_elementwise.py``, 1/4 NaN + 1/4 +Inf + 1/4 -Inf + 1/4
randn-finite input mix) is still the authoritative numeric check for this risk on Ascend;
round_1_result.md records its outcome.

**Tiling -- no chunking needed, unlike the schedule's two candidates**: since the cast
body never requests a vectorized bool store, ``threads=min(2048, tile)`` never hits the
bool-store vectorization wall at any tile width tried (``round_1_schedule.md``'s
Candidate A/B chunking machinery is dead code for this body and is not implemented here).
The initial uniform ``tile=8192`` (the shape every sibling unary op's tuning history
assumed) compiled and passed M3 cleanly but missed all 5 latency targets by a wide margin
(~53-61% of best_baseline, vs the 80% bar) -- a bandwidth-bound shortfall, not a
correctness defect. Two widening passes (the same lever that fixed ``RsqrtFwdOp``'s
DMA-bound miss, dev_process.md §7.5) recovered real but incomplete margin: uniform
``tile=16384`` -> ~59-70%; the final shipped **per-dtype** tile -- ``32768`` for fp16/bf16
(192 KiB/block at 2 stages) and ``16384`` for fp32 (160 KiB/block; fp32 can't follow
fp16/bf16 to 32768 without a 320 KiB footprint, over ``BitwiseNotFwdOp``'s confirmed
256 KiB overflow point) -- -> ~62-75%, closest at the two 256M cases. All 5 cases still
miss the 80% bar at the end of round 1; round_1_result.md has the full iteration table and
a concrete round-2 starting point (further `num_stages`/tile sweep, or an
`msprof_detail`-based pipe-occupancy diagnosis, neither attempted this round).

**2026-10-20 pack branch**: only 16-bit float shapes exactly divisible by
``4*num_blocks*tile_packed`` route to ``_make_isfinite_word32``, which reads two native
int32 words (four 16-bit values) per output word, evaluates the approved IEEE
exponent-field predicate (``(raw_bits & exp_mask) != exp_mask``) with nonnegative typed
masks, and stores the four 0/1 results in bytes 0..3 of one int32 word (little endian);
``forward`` unpacks with ``.view(torch.uint8).view(torch.bool)``. The isolated word32
prototype passed both 16-bit full 65,536 raw code spaces with CPU and device dual
oracles, raw-transfer, canonical bytes and fresh-output gates, and measured 18.2503us vs
20.5978us for the per-half mask body on 16M-fp16. Float32 is not in ``_PACK_CONFIG``:
its 88fa full M4 showed the packed path slower than the original scalar persistent path,
and the reviewed thread comparison showed only ~1.4% gain there. Every other shape
(small, odd/tail, non-divisible) and all six non-float dtypes keep the pre-existing
paths above unchanged.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a
CUDA-only process) is a no-op: the Ascend-specific imports are guarded by one
``try/except ImportError`` (``_ASCEND_IMPORT_ERROR``), re-raised lazily only when
``AscendIsfiniteKernel.__init__`` is actually reached, mirroring every sibling Ascend
kernel file in this directory.

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

__all__ = ["AscendIsfiniteKernel"]

# The 3 float dtypes that can represent NaN/Inf: the only ones a real TileLang program is
# built for. Defined locally (plain tuple of torch.dtype) rather than imported from
# kernels/elementwise/math_unary.py, keeping this file's only cross-package dependency the
# tilelang.ascend try/except block (round_1_schedule.md §3.1).
FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The other 6 manifest dtypes: none of them can represent NaN or an infinity, so an
# int/bool input is *always* finite. forward() answers them directly with
# torch.ones(..., dtype=torch.bool) -- no TileLang program at all. This is the
# True/ones constant (round_1_schedule.md §3.1) -- the opposite of IsnanFwdOp's/
# IsinfFwdOp's False/zeros fallback for the same 6 dtypes.
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
ALWAYS_TRUE_DTYPES = INT_DTYPES + (torch.bool,)

# Tile size and stage count for the persistent, bandwidth-bound path (module docstring):
# fixed across all 3 float dtypes -- the asymmetric bool output means this op moves
# strictly less traffic per element than a same-dtype-in/out unary op at the same tile
# geometry, so there is UB headroom to spare without per-dtype tile scaling. No chunking
# of the per-lane compute loop is needed for the chosen cast-based body (module docstring).
# Round-1 M4 lever (module docstring "Tiling" section): the initial uniform tile=8192 missed
# all 5 latency targets by a wide margin (~55-61% of best_baseline, vs the 80% bar) even
# though M3 compiled/passed cleanly -- a bandwidth-bound shortfall, not a correctness bug.
# Widening the tile (the same lever that fixed RsqrtFwdOp's DMA-bound miss,
# dev_process.md §7.5) to 16384 uniformly recovered some margin (~59-70%) but still missed
# all 5. A further per-dtype widening -- 32768 for fp16/bf16 (192 KiB/block at 2 stages),
# 16384 for fp32 (160 KiB/block; 32768 would be 320 KiB, over BitwiseNotFwdOp's confirmed
# 256 KiB overflow point for a 4-byte-wide input) -- is the tile this file ships. See
# round_1_result.md for the measured outcome of each step.
_TILE_FP16_BF16 = 32768
_TILE_FP32 = 16384
_NUM_STAGES = 2

# 2026-10-20 pack branch (isolated prototype first): one int32 output word consumes four
# native float values, evaluates the real T.isfinite predicate on each after promotion to
# fp32, and stores the four 0/1 results in bytes 0..3 of that word (little endian, the same
# byte layout the reviewed LogicalNot float-pack template uses; that template's zero
# predicate and input-bit reinterpretation are NOT reused). tile_packed is measured in
# int32 words. UB footprint is num_stages*tile_packed*(4*input_itemsize + 4) bytes:
# 192 KiB for 16-bit P=8192, 160 KiB for fp32 P=4096, both under the 216 KiB budget.
# Entries are (num_blocks, tile_packed, num_stages, threads). 2026-10-20 detailed
# same-body thread comparison on 16M cases: 1024 reduce duration/AIV cycles by ~12%
# for float16/bfloat16, but only ~1.4% for float32. The 88fa full M4 also showed the
# float32 packed path slower than the original scalar persistent path, so float32 is
# intentionally absent here and keeps the pre-existing scalar routing. Only
# exact-divisible large 16-bit float shapes reach the packed path; every other shape
# keeps the original implementation.
_PACK_CONFIG = {
    "float16": (64, 8192, 2, 1024),
    "bfloat16": (64, 8192, 2, 1024),
}
# IEEE exponent field per packed dtype, for the isolated-and-approved mask predicate:
# finite iff the exponent field is not all ones (the mantissa only splits Inf vs NaN).
_MASK_BITS = {
    "float16": ("int16", 0x7C00),
    "bfloat16": ("int16", 0x7F80),
    "float32": ("int32", 0x7F800000),
}
# word32-input pack4: two native 16-bit values share one int32 word; low/high
# exponent masks. The isolated prototype passed both 16-bit full code spaces and
# measured 18.2503us vs 20.5978us for the per-half mask body on 16M-fp16.
_WORD32_MASKS = {
    "float16": (0x7C00, 0x7C000000),
    "bfloat16": (0x7F80, 0x7F800000),
}
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
    the nearest power of two, rather than hardcoding the figure outright. Every prior
    Ascend elementwise kernel with this op's one-load/one-store-per-element cost shape
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


if T is not None:

    @T.macro
    def _isfinite_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = isfinite(x_ub[:n])`` as bool, compiler-vectorized over ``threads``.

        ``T.isfinite(T.cast(x, "float32"))`` -- matching the in-tree CUDA
        ``IsfiniteFwdKernel.op_func`` exactly -- unconditionally for all 3 float dtypes:
        mandatory for correctness (a bare ``T.isfinite(x)`` fails host-side for bfloat16,
        and fails Bisheng codegen for float16 via a transitive undeclared-``hfabs``
        gap inherited from ``isfinite``'s internal ``abs()`` call for 16-bit types --
        module docstring / round_1_schedule.md §4), and confirmed by a live compile probe
        to compile clean at the real ``tile=8192``/``threads=2048`` persistent-grid shape
        with **no** bool-store vectorization failure, unlike the schedule's original
        algebraic-identity candidate (module docstring).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable`` under TileLang's eager builder.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.isfinite(T.cast(x_ub[i], "float32"))

    @T.macro
    def _isfinite_pack4_body(x_ub, y_ub, n_packed: int, bits_dtype: str,
                             exp_mask: int, threads: int) -> None:
        """``y_ub[i] = pack4(isfinite(x_ub[4*i : 4*i+4]))``; lane k -> byte k, little endian.

        The predicate is the approved IEEE exponent-field test: finite iff
        ``(raw_bits & exp_mask) != exp_mask`` (all-ones exponent means Inf or NaN;
        the mantissa only distinguishes them, so +-0/subnormal/finite/inf/nan
        classification is exact). The raw bits come from ``T.reinterpret`` of the
        native dtype, with no fp32 promotion, fabs or NaN self-compare. The isolated
        prototype passed both 16-bit full 65536 code spaces and fp32 specials/random
        with CPU/device dual oracles, raw-transfer, canonical bytes and fresh-output
        gates before this production body replaced the T.isfinite chain. A ``@T.macro``
        for the same eager-builder reason as ``_isfinite_body`` above.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n_packed):
                b0 = T.cast(
                    T.bitwise_and(T.reinterpret(x_ub[4 * i + 0], bits_dtype),
                                  T.cast(exp_mask, bits_dtype))
                    != T.cast(exp_mask, bits_dtype), "int32")
                b1 = T.cast(
                    T.bitwise_and(T.reinterpret(x_ub[4 * i + 1], bits_dtype),
                                  T.cast(exp_mask, bits_dtype))
                    != T.cast(exp_mask, bits_dtype), "int32")
                b2 = T.cast(
                    T.bitwise_and(T.reinterpret(x_ub[4 * i + 2], bits_dtype),
                                  T.cast(exp_mask, bits_dtype))
                    != T.cast(exp_mask, bits_dtype), "int32")
                b3 = T.cast(
                    T.bitwise_and(T.reinterpret(x_ub[4 * i + 3], bits_dtype),
                                  T.cast(exp_mask, bits_dtype))
                    != T.cast(exp_mask, bits_dtype), "int32")
                lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
                hi = T.bitwise_or(
                    T.shift_left(b2, T.cast(16, "int32")),
                    T.shift_left(b3, T.cast(24, "int32")),
                )
                y_ub[i] = T.bitwise_or(lo, hi)

    @T.macro
    def _isfinite_word32_body(a_ub, c_ub, n_packed: int, low_mask: int,
                              high_mask: int, threads: int) -> None:
        """Two int32 input words -> four 16-bit exponent predicates -> one packed word.

        Each input word holds two native 16-bit values (little endian); low_mask and
        high_mask are both nonnegative and clear the sign bit, so the masked value is
        a safe nonnegative int32 compared exactly against the typed exponent mask.
        The four 0/1 results are stored in bytes 0..3 with the unchanged pack shifts.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n_packed):
                w0 = a_ub[2 * i]
                w1 = a_ub[2 * i + 1]
                b0 = T.cast(
                    T.bitwise_and(w0, T.cast(low_mask, "int32"))
                    != T.cast(low_mask, "int32"), "int32")
                b1 = T.cast(
                    T.bitwise_and(w0, T.cast(high_mask, "int32"))
                    != T.cast(high_mask, "int32"), "int32")
                b2 = T.cast(
                    T.bitwise_and(w1, T.cast(low_mask, "int32"))
                    != T.cast(low_mask, "int32"), "int32")
                b3 = T.cast(
                    T.bitwise_and(w1, T.cast(high_mask, "int32"))
                    != T.cast(high_mask, "int32"), "int32")
                lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
                hi = T.bitwise_or(
                    T.shift_left(b2, T.cast(16, "int32")),
                    T.shift_left(b3, T.cast(24, "int32")),
                )
                c_ub[i] = T.bitwise_or(lo, hi)

    @functools.lru_cache(maxsize=64)
    def _make_isfinite_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (module docstring).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_isfinite_kernel``) only reaches this builder for a shape that does.
        Serves all 5 benchmarked manifest cases and the 1,048,576-element correctness
        smoke fixture.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_isfinite_persistent requires n_total ({n_total}) to divide "
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
                    _isfinite_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_isfinite_packed(n_total: int, dtype: str, num_blocks: int,
                              tile_packed: int, num_stages: int, threads: int):
        """Persistent packed grid: four input values per int32 output word.

        ``tile_in = 4*tile_packed``; ``n_total`` must divide
        ``num_blocks*tile_in`` exactly (the routing predicate below checks that before
        calling). All six scheduling parameters are part of the lru-cache key, so the
        tuner's compile_fn reuses the exact same builder the factory elaborates.
        """
        tile_in = 4 * tile_packed
        block_span = num_blocks * tile_in
        assert n_total % block_span == 0, (
            f"_make_isfinite_packed requires n_total ({n_total}) to divide "
            f"num_blocks*tile_in ({block_span}) exactly; routing keeps other shapes "
            "on the original implementation"
        )
        iterations = n_total // block_span
        bits_dtype, exp_mask = _MASK_BITS[dtype]

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total // 4,), "int32")):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_in,), dtype)
                y_ub = T.alloc_shared((tile_packed,), "int32")
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin_in = (step * num_blocks + bx) * tile_in
                    begin_out = (step * num_blocks + bx) * tile_packed
                    T.copy(X[begin_in : begin_in + tile_in], x_ub)
                    _isfinite_pack4_body(x_ub, y_ub, tile_packed, bits_dtype,
                                         exp_mask, threads=threads)
                    T.copy(y_ub, Y[begin_out : begin_out + tile_packed])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_isfinite_word32(n_total: int, dtype: str, num_blocks: int,
                              tile_packed: int, num_stages: int, threads: int):
        """Persistent word32-input pack4 builder for the two 16-bit dtypes.

        The host passes ``input.view(int32)`` (``n_total//2`` words). Each output
        int32 word consumes two input words and packs four exponent predicates.
        Resources stay ``num_stages*tile_packed*(2*4 + 4)`` bytes: 192 KiB for
        P=8192/stage2, unchanged from the per-half mask body.
        """
        tile_words = 2 * tile_packed
        input_words = n_total // 2
        assert input_words % (num_blocks * tile_words) == 0, (
            f"_make_isfinite_word32 requires input_words ({input_words}) to divide "
            f"num_blocks*tile_words ({num_blocks * tile_words}) exactly"
        )
        iterations = input_words // (num_blocks * tile_words)
        low_mask, high_mask = _WORD32_MASKS[dtype]

        @T.prim_func
        def main(A: T.Tensor((input_words,), "int32"),
                 C: T.Tensor((n_total // 4,), "int32")):
            with T.Kernel(num_blocks) as bx:
                a_ub = T.alloc_shared((tile_words,), "int32")
                c_ub = T.alloc_shared((tile_packed,), "int32")
                T.annotate_buffer_versions({a_ub: num_stages, c_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin_in = (step * num_blocks + bx) * tile_words
                    begin_out = (step * num_blocks + bx) * tile_packed
                    T.copy(A[begin_in : begin_in + tile_words], a_ub)
                    _isfinite_word32_body(a_ub, c_ub, tile_packed, low_mask,
                                          high_mask, threads=threads)
                    T.copy(c_ub, C[begin_out : begin_out + tile_packed])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _packed_shape(n_total: int, dtype: str) -> bool:
        """True only when the packed builder divides this exact shape and dtype."""
        if dtype not in _PACK_CONFIG:
            return False
        blocks, tile_packed, _stages, _threads = _PACK_CONFIG[dtype]
        return n_total % (4 * blocks * tile_packed) == 0

    @functools.lru_cache(maxsize=64)
    def _make_isfinite_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not
        evenly divide: one block, a main loop over full ``tile``-sized chunks plus one
        smaller, statically-sized tail chunk. Mirrors ``relu.py``'s
        ``_make_relu_generic_single_block``/``isnan.py``'s
        ``_make_isnan_generic_single_block`` verbatim in shape, with ``y_ub``/``Y``
        declared ``"bool"`` and the body swapped for ``_isfinite_body``. Handles the two
        small, non-benchmarked correctness shapes this op's test surface needs (4096 and
        64 elements, neither reached by any benchmarked manifest workload) -- both
        confirmed by live compile probe to compile clean at all 3 float dtypes with no
        bool-store vectorization issue (the tail chunk's ``threads=min(128, remainder)``
        gives up to 32 elements/thread, still scalar-store under the cast-based body).
        Not performance-tuned.
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
                        _isfinite_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), "bool")
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _isfinite_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _tile_for_dtype(dtype: str) -> int:
        """Per-dtype tile width (module docstring): 32768 for fp16/bf16 (192 KiB/block at
        2 stages), 16384 for fp32 (160 KiB/block -- 32768 would be 320 KiB, over
        BitwiseNotFwdOp's confirmed 256 KiB overflow point for a 4-byte-wide input).
        """
        return _TILE_FP32 if dtype == "float32" else _TILE_FP16_BF16

    def _make_isfinite_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements of ``dtype``: persistent grid when it
        divides ``num_blocks * tile`` exactly, else the generic single-block fallback.
        """
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")
        if _packed_shape(n_total, dtype):
            blocks, tile_packed, stages, threads = _PACK_CONFIG[dtype]
            return _make_isfinite_word32(
                n_total, dtype, blocks, tile_packed, stages, threads)
        num_blocks = _calibrated_num_blocks()
        tile = _tile_for_dtype(dtype)
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_isfinite_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_isfinite_generic_single_block(n_total, dtype, tile)


class AscendIsfiniteKernel(Kernel, UnaryPredicateFwdInterface):
    """IsFinite on Ascend NPUs: ``y = isfinite(x)``, a new contiguous ``torch.bool`` tensor.

    Implements ``UnaryPredicateFwdInterface`` (**not** ``UnaryElementwiseFwdInterface`` --
    see this module's docstring for why that distinction is load-bearing for
    ``IsfiniteFwdOp``'s own ``interfaces`` override). ``devices`` is NPU only;
    ``supported_archs`` is left ``None`` -- unlike the CUDA ``UnaryKernel``/
    ``FloatUnaryKernel`` family this class does not inherit, there is no Ascend SoC-
    version gate here to replace the CUDA SM-version one (``Kernel._check_arch`` skips
    the check entirely when ``supported_archs is None``).

    Accepts the op's full 9-dtype manifest domain, not just the 3 floats this kernel
    compiles a TileLang program for (module docstring): ``refusal()`` returns ``None``
    unconditionally, and ``forward()`` branches internally -- the 3 float dtypes
    (``float16``, ``bfloat16``, ``float32``) run the compiled TileLang program below; the
    other 6 (``bool`` plus the 5 ``INT_DTYPES``, none of which can represent NaN or an
    infinity, hence always finite) are answered directly with
    ``torch.ones(input.shape, dtype=torch.bool, device=input.device)``, bit-identical to
    the CUDA ``AlwaysTrueFwdKernel.handler`` and built with no TileLang compile at all.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        # Accepts the FULL manifest domain (9 dtypes), not just the 3 floats this kernel
        # actually compiles a program for -- see module docstring / round_1_schedule.md
        # §3.2: a narrow refusal() here would leave npu+int/bool calls with NO available
        # kernel at all (this kernel refuses, and AlwaysTrueFwdKernel/IsfiniteFwdKernel
        # are cuda-only and refuse on device too), breaking
        # test_predicate_non_float_constant and
        # test_single_tensor_op_completes_every_declared_dtype on npu.
        return None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendIsfiniteKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._use_pack = (
            dtype in FLOAT_DTYPES and _packed_shape(n_total, _dtype_str(dtype))
        )
        self._compiled_fn = (
            _make_isfinite_kernel(n_total, _dtype_str(dtype)) if dtype in FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isfinite(input)`` elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements on an NPU device, contiguous in whatever
                shape the caller passed, in ``self.dtype``.

        Returns:
            A new contiguous ``torch.bool`` tensor on the same device, shaped as
            ``input``. For the 6 non-float manifest dtypes (none of which can represent
            NaN or an infinity), this is always all-``True`` -- the constant-output path,
            no TileLang kernel involved (module docstring).
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if self.dtype not in FLOAT_DTYPES:
            return torch.ones(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(
            flat_input.view(torch.int32) if self._use_pack else flat_input)
        if self._use_pack:
            # The word32 kernel stores packed 0/1 bytes in an int32 buffer; expose the
            # same torch.bool contract the scalar path returns.
            return flat_output.view(torch.uint8).view(torch.bool).reshape(input.shape)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("IsfiniteFwdOp", "ascend", AscendIsfiniteKernel)
except ImportError:
    pass
