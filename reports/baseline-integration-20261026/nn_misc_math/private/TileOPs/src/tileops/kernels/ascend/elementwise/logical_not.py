"""Ascend TileLang kernel for ``LogicalNotFwdOp``: elementwise ``not x``, bool output.

Design: ``docs/ascend950/dev/LogicalNotFwdOp/round_1_schedule.md`` (Planner, round 1).
Pure bandwidth-bound unary predicate (load a tile, compare/XOR it in UB, store a bool
tile back), no Cube engine involved.

**One kernel class, not the CUDA-side two-class split.** The in-tree CUDA
implementation (``tileops/kernels/elementwise/logical.py``) splits this op into
``LogicalNotFwdKernel`` (``x == 0`` for all 9 manifest dtypes, including bool) and
``LogicalNotBoolStorageFwdKernel`` (a uint8-view XOR-1 shortcut CUDA prefers for bool,
via ``preferred_over``). The schedule's read of ``_base.py`` confirms the split is a
CUDA-only performance shortcut -- CUDA has no vectorized bool type -- not a correctness
requirement; ``x == T.cast(0, x.dtype)`` is already mathematically correct for bool on
this backend (``BitwiseNotFwdOp`` round 1 ran the literal expression
``v == T.cast(0, "bool")`` on real hardware and got M3 PASS, round_1_schedule.md §3/§4).
So this file registers exactly one Ascend implementation,
``AscendLogicalNotKernel(Kernel, UnaryPredicateFwdInterface)``, with two strategies
picked internally per dtype -- not two separate ``Kernel`` subclasses and not two
``register_kernel_type`` calls.

**Strategy 1 -- generic compare-to-zero** (uint8/int8/int16/int32/int64, plus bool when
strategy 2 is unavailable; **round 2 narrowed this list -- see Strategy 3 below for why
float16/bfloat16/float32 moved out of it**): ``c[i] = (a[i] == T.cast(0, dtype_str))``.
The output buffer is always ``bool`` (1 byte), independent of the input dtype -- a
combination no sibling op in this directory has needed before (every prior unary op
here has ``output.dtype == input.dtype``). Per round_1_schedule.md §5.1, this strategy
does not bet on "vectorized load, scalar store" decoupling: every dtype under it runs
at exactly 1 element/thread (``tile == threads``), which keeps both the load and the
bool store scalar and sidesteps the sub-32-bit vectorized-store rejection
(``BitwiseNotFwdOp`` round 1: "SimtVF cannot convert vector type int8x4/uint8x4/int16x4
to an Ascend type").

**Round 1 note, resolved in round 2**: round 1 flagged that this project's
``IsnanFwdOp`` Ascend kernel independently reports a 4-elements/thread float-in/bool-out
kernel compiling at ``tile=8192`` -- i.e. a precedent for the mixed vectorized-
load/scalar-store combination this op's own body could not get through at the same
tile. Round 1's own direct attempt (same tile=8192, this op's literal
``a[i] == T.cast(0, dtype_str)`` body) reproduced the failure
(``SimtVF cannot convert vector type int8x4/int8x2 to an Ascend type`` for
float16/bfloat16/float32 respectively) -- confirmed a real discrepancy between the two
kernels' bodies, not a fluke, and recorded as an *unresolved, project-wide* open
question in dev_process.md §7.5 ("Batch 5 收尾新增"): the one structural difference
anyone had identified was that ``IsnanFwdOp`` promotes to float32 before its predicate
(``T.isnan(T.cast(x, "float32"))``) while this op compares the native dtype directly
against a same-dtype zero (``a[i] == T.cast(0, dtype_str)``) -- an untested hypothesis,
not a confirmed mechanism, per round 1's own result doc.

**Round 2 resolved this with a real experiment, and the hypothesis did not pan out**: a
compile-only probe (no NPU touched) held the kernel geometry fixed at the exact
failing configuration (``num_blocks=64, tile=8192, threads=2048``) and varied only the
compute body -- (A) native-dtype compare (round 1's literal body) vs. (B) promote to
float32 then compare (``T.cast(a[i], "float32") == T.cast(0, "float32")``, the
mechanical analogue of ``IsnanFwdOp``'s promotion). **Both (A) and (B) failed
identically** (``int8x4`` for float16/bfloat16, ``int8x2`` for float32, byte-for-byte
the same error round 1 already saw) -- promoting to fp32 before the compare does
*not* avoid the vectorized bool-store rejection for this op's body shape. This
falsifies the "native vs. promote" hypothesis as *the* mechanism (at least as the sole
explanatory variable) for this op, and leaves the "expression-tree complexity" family
of hypotheses from the Isnan/Isinf/Isfinite trio (dev_process.md §7.5) looking more
likely: ``IsnanFwdOp``'s compute is a single builtin predicate call
(``T.isnan(...)``) lowered by the compiler as one unit, whereas this op's body is an
explicit ``==`` comparison against a literal-zero cast -- a different expression shape
even after promotion. This project-wide question is reported here with real evidence
either way (round_2_result.md has the full probe transcript) rather than resolved by
further guessing.

**Strategy 2 -- bool 4-pack XOR** (bool input only, round_1_schedule.md §5.2, the
round's primary optimization target): reinterpret 4 bool elements as 1 int32 word
(``x.view(torch.uint8).view(torch.int32)``), XOR every word against the constant
``0x01010101``, then reinterpret the result back to bool. A PyTorch bool tensor's
storage invariant is that every byte is exactly ``0x00`` or ``0x01``; XOR has no
cross-bit or cross-byte dependency (unlike, say, addition's carry chain), so grouping
4 bytes into one XOR instead of 4 independent ones is bit-exact -- the same argument
``BitwiseNotFwdOp`` already used for its int64-as-2x-int32 fallback. This turns the bool
path into an ordinary int32 XOR-by-constant kernel, which reuses the int32 tiling
(``num_blocks<=64, tile=8192, threads=2048``, 4 elements/thread, ``int32x4`` vector
store) ``BitwiseNotFwdOp``/``NegFwdOp`` already validated on real hardware. Verified
bit-exact against ``torch.logical_not`` on CPU tensors (n=64, 2**20, 2**24) before any
TileLang code was written (round_1_schedule.md §8 item 2). Falls back to strategy 1
(scalar bool compare) when the input's element count is not a multiple of 4 -- not a
case any known manifest workload or test fixture hits, but cheap defensiveness since
``.view(torch.int32)`` would otherwise raise.

**Strategy 3 -- float 4-lane pack-compare (round 2, float16/bfloat16/float32 only)**:
this is exactly the "round 2 candidate" round_1_schedule.md §5.3 anticipated and
shelved -- round 1 shipped Strategy 1 (1 element/thread) for the float dtypes and
measured ``16M-float16``/``16M-float32`` at only 51.2%/59.0% of the 80% target
(round_1_result.md §5); round 1's own in-round attempt to fix this by simply widening
Strategy 1's tile to 8192 hit the ``int8x4``/``int8x2`` compile failure this file's
docstring discusses above, and round 2's compile probe ruled out "promote to fp32
first" as a one-line fix for the same failure (also discussed above). So round 2 takes
the schedule's original, more involved path instead: unlike Strategy 2, there is no
free bit-reinterpret available here -- an arbitrary float's bit pattern is not limited
to ``{0, 1}``, so XOR-ing raw bits would not compute a comparison. 4 lanes must be
**genuinely, independently compared** to zero, then their 0/1 results assembled by hand
into one ``int32`` word before a single wide store:

```python
b0 = T.cast(a_ub[4 * i + 0] == T.cast(0, dtype_str), "int32")
b1 = T.cast(a_ub[4 * i + 1] == T.cast(0, dtype_str), "int32")
b2 = T.cast(a_ub[4 * i + 2] == T.cast(0, dtype_str), "int32")
b3 = T.cast(a_ub[4 * i + 3] == T.cast(0, dtype_str), "int32")
c_ub[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
```

Lane ``k`` of each group of 4 floats lands in byte ``k`` of the packed word (lane 0 in
the low byte, no shift; lane 3 in the high byte, ``<<24``) -- the same little-endian
byte layout ``forward()``'s ``.view(torch.uint8).view(torch.bool)`` unpack already
relies on for Strategy 2's bool path, verified bit-exact on the CPU host (no NPU) for
float16/bfloat16/float32 inputs, including ``0.0``/``-0.0``/``NaN``/``+-inf``, against
``torch.logical_not`` before any TileLang code for this strategy was written
(round_2_schedule.md). The packed ``int32`` output buffer is never narrower than 32
bits, so this sidesteps the sub-32-bit vectorized-store rejection entirely -- unlike
Strategy 1, there is no bool (or other sub-32-bit) buffer anywhere in the compute body,
only a plain ``int32``-producing kernel structurally identical to
``BitwiseNotFwdOp``/``NegFwdOp``'s already-validated int32 tiling, the same reuse
argument Strategy 2 already made for bool. Verified by a second, geometry-matching
compile probe (round_2_schedule.md) at the exact tile/thread configuration this file
ships (``_FLOAT_PACK_TILE``, below) before being wired into the real kernel.

Per-dtype packed tile width is **not** uniform like Strategy 1's ``_GENERIC_TILE``
because this strategy's UB footprint carries two buffers of different widths (the float
input tile, and the ``int32`` packed-output tile which has 1/4 as many elements but
always 4 bytes/element) against this SKU's ``216 KiB`` (``221184``-byte) UB budget
(dev_process.md §7.5, the hardware-validated figure ``SigmoidFwdOp``/``SiluFwdOp``/
``ExpFwdOp`` already confirmed): float16/bfloat16 (2 bytes/input-element) afford a
packed tile of 8192 ``int32`` words (32768 input elements/block; UB =
``2 stages x (32768x2 + 8192x4) = 196608 B = 192 KiB``, matching Strategy 2's
already-validated ``int32x4``-store geometry exactly); float32 (4 bytes/input-element)
is halved to a packed tile of 4096 (16384 input elements/block; UB =
``2 stages x (16384x4 + 4096x4) = 163840 B = 160 KiB``, an ``int32x2`` store -- not the
``int32x4`` width Strategy 2 uses, but still a native 32-bit-or-wider store, a width
this project has never seen rejected). Falls back to Strategy 1 when the input's
element count is not a multiple of 4 -- like Strategy 2, not a case any known manifest
workload or test fixture hits, but cheap defensiveness for the same reason.

**Grid sizing, generalized beyond what round_1_schedule.md worked out by hand.** The
schedule's own worked examples (§5.2's table, §6's n=64 case) only cover the two
benchmarked bool cases and the one known-tiny correctness shape. Reconciling them
exposed a case neither section's formula quite covers: ``test_logical.py``'s
own M3 correctness fixture (``n_total=1,048,576``, the file's "authoritative M3
threshold" per round_1_schedule.md §7) packs to 262,144 int32 elements under strategy
2 -- smaller than one full ``num_blocks=64, tile=8192`` group (524,288), yet larger
than one tile, so neither "run the full persistent grid" nor "single tiny scalar
block" (the schedule's literal n=64 formula, which would force 32 elements/thread here
-- an untested vector width) is both correct and safe. ``_choose_grid`` below picks a
third option already used by this op's own benchmarked cases' spirit: keep the
validated tile/thread geometry and only shrink ``num_blocks`` (262,144 / 8192 = 32
blocks, 1 iteration) when ``n`` divides the tile but not the full group. This keeps
every strategy-2 shape at exactly 4 elements/thread (the one validated int32 vector
width) and only the true small-shape case (strategy 1's n=64, strategy 2's packed
n=16) drops to a single untiled, unpipelined block.
"""

from functools import lru_cache
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # no Ascend-aware tilelang in this process: safe no-op import
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendLogicalNotKernel"]

# Persistent-grid defaults shared by both strategies, carried over from
# BitwiseNotFwdOp/NegFwdOp round 1 (this SKU's real vector core count is 72, not a power
# of two; every n_total this op sees -- raw or packed -- is a pure power of two, so 72
# cannot divide any of them and 64 is the nearest feasible, already-validated value;
# round_1_schedule.md §1/§8 item 5).
_NUM_BLOCKS = 64
_NUM_STAGES = 2

# Strategy 1 (generic compare-to-zero): tile == threads == 2048 for every dtype, i.e.
# exactly 1 element/thread. Deliberately not dtype-scaled the way BitwiseNotFwdOp scales
# tile by element width -- here the constraint is the *output* dtype (always 1-byte
# bool), not the input, and it applies uniformly regardless of input width (module
# docstring, round_1_schedule.md §5.1). **Round 2 narrowed this strategy's real
# domain to uint8/int8/int16/int32/int64 (plus bool when Strategy 2 is unavailable)**:
# the three float dtypes moved to Strategy 3 below; see that strategy's docstring and
# `_FLOAT_PACK_TILE` for why. This constant and `_cmp_zero_body` are kept exactly as
# round 1 shipped them -- M3 already validated every non-float dtype against this tile,
# and none of them are in the M4 benchmark scope, so there is no reason to touch this
# path in round 2.
#
# Round 1 tried and reverted an 8192-element/4-elements-per-thread tile for
# float16/bfloat16/float32 only, modeled on this project's ``IsnanFwdOp`` Ascend kernel
# (``isnan.py``), whose docstring reports a compile probe where an asymmetric
# float-in/bool-out kernel's bool *store* stays scalar even at 4 elements/thread. On
# real hardware, this op's own comparison body (``c[i] = a[i] == T.cast(0, dtype_str)``)
# did not reproduce that: all three float dtypes failed to compile at tile=8192 with
# ``SimtVF cannot convert vector type int8x4/int8x2 to an Ascend type``. Round 2's
# compile probe went further and showed promoting to float32 before the compare does
# *not* avoid this either (both variants failed identically at this tile) -- see the
# module docstring's "Round 1 note, resolved in round 2" for the full probe result.
# Round 2 solves the float16/bfloat16/float32 M4 miss with Strategy 3 (manual 4-lane
# pack-into-int32) instead of trying to widen this strategy's tile further.
_GENERIC_TILE = 2048

# Strategy 2 (bool 4-pack XOR): the int32 tiling BitwiseNotFwdOp/NegFwdOp already
# validated (4 elements/thread, int32x4 vector store, 128 KiB UB footprint for
# 2 buffers x 2 stages x 8192 elements x 4 bytes -- round_1_schedule.md §5.2).
_PACK_TILE = 8192

# Unsigned zero-byte word constants (schedule 2026-10-20 conditional candidate):
# each low-seven-bit byte plus 0x7f is at most 0xfe, so the addition cannot carry
# across a byte boundary.  OR with the original word preserves each byte's high
# bit, OR with 0x7f forces the low seven bits on, complement exposes exactly the
# original zero bytes, and a *logical* shift by 7 places that high bit at each
# byte's bit 0.  Masking with 0x01010101 yields canonical 0/1 bytes.
_ZERO_WORD_LOW7 = 0x7F7F7F7F
_ZERO_WORD_HIGH = 0x01010101

# Strategy 3 (round 2, float 4-lane pack-compare): per-dtype packed tile width, in
# *packed int32 elements* (each packed word consumes 4 input elements). Not uniform
# across dtypes -- see the module docstring's Strategy 3 section for the UB-budget
# arithmetic against this SKU's confirmed 221184-byte (216 KiB) budget
# (dev_process.md §7.5). Both the builder (to size the kernel) and this docstring's
# footprint argument refer to this dict, hence module-level.
_FLOAT_PACK_TILE = {
    "float16": 8192,  # 32768 input elements/block; UB = 2*(32768*2+8192*4) = 192 KiB
    "bfloat16": 8192,  # same arithmetic as float16 (2 bytes/element)
    "float32": 4096,  # 16384 input elements/block; UB = 2*(16384*4+4096*4) = 160 KiB
}


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _choose_grid(n: int, tile: int, num_blocks_default: int, threads_cap: int = 2048):
    """Pick ``(num_blocks, tile, threads, iterations, single_block)`` for ``n`` elements.

    Three tiers, in order:

    1. ``n < tile``: one untiled, unpipelined block covering exactly ``n`` elements,
       ``threads = min(n, threads_cap)``. The only tier that can leave ``tile != threads``
       in a way that is *not* a multiple of ``threads_cap``'s validated vector widths --
       but ``n < tile`` only arises this round for already-scalar shapes (``n <=
       threads_cap``), so this stays 1 element/thread in every case this round actually
       builds.
    2. ``n % tile == 0`` (always true when ``n >= tile`` and both are powers of two, which
       is every shape this round's manifest/tests use): a persistent grid with
       ``num_blocks = min(num_blocks_default, n // tile)`` blocks, each doing
       ``iterations = (n // tile) // num_blocks`` ``T.Pipelined`` passes. Shrinking
       ``num_blocks`` (rather than forcing the full ``num_blocks_default`` and failing to
       divide) is what lets this serve a shape smaller than one full
       ``num_blocks_default * tile`` group -- e.g. this op's own bool-pack M3 fixture at
       262,144 packed elements, below the 524,288-element full group -- without changing
       the per-thread vectorization width.
    3. Anything else (``n >= tile`` but not a multiple of it): not reachable by any
       manifest workload or test fixture this round knows about (every known ``n`` is a
       power of two at least as large as ``tile``, or already caught by tier 1). Raises
       rather than silently computing a wrong answer, matching ``BitwiseNotFwdOp``'s
       "no remainder tile" policy.

    Returns:
        A 5-tuple ``(num_blocks, tile, threads, iterations, single_block)``.
    """
    if n <= 0:
        raise ValueError(f"_choose_grid: n must be positive, got {n}")
    if n < tile:
        return 1, n, min(n, threads_cap), 1, True
    if n % tile == 0:
        num_full = n // tile
        num_blocks = min(num_blocks_default, num_full)
        iterations = num_full // num_blocks
        return num_blocks, tile, min(tile, threads_cap), iterations, False
    raise ValueError(
        f"_choose_grid: n={n} is >= tile={tile} but does not divide it; no remainder-tile "
        "handling this round (every known manifest/test shape is a power of two that "
        "either is smaller than tile or divides it exactly)"
    )


if T is not None:

    @T.macro
    def _cmp_zero_body(a_ub, c_ub, n: int, dtype_str: str, threads: int) -> None:
        """``c_ub[:n] = (a_ub[:n] == 0)`` as bool. ``@T.macro``, not a plain function --

        an undecorated helper's ``for i in T.Parallel(n):`` is not iterable under
        TileLang's eager builder (confirmed precedent: ``isnan.py``'s ``_isnan_body``
        docstring).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                c_ub[i] = a_ub[i] == T.cast(0, dtype_str)

    @T.macro
    def _pack_zero_word_body(a_ub, c_ub, n: int, threads: int) -> None:
        """``c_ub[:n] = canonical_byte01(input word == 0 per byte)``.

        Unsigned u formula (schedule 2026-10-20):
        ``z = ((~(((u & 0x7f7f7f7f) + 0x7f7f7f7f) | u | 0x7f7f7f7f)) >> 7) & 0x01010101``.
        Addition, complement and shift are all uint32 operations; the result is
        reinterpreted as int32 only for the packed store.  Each low-seven-bit byte
        plus 127 is <= 254, so there is no inter-byte carry; OR with u catches each
        original high bit.  This gives canonical byte 1 exactly for original byte
        zero, covering all 256 storage bytes (unlike XOR 0x01010101, which yields
        0x03 for input byte 0x02 and therefore the wrong truth value).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                u = T.reinterpret(a_ub[i], "uint32")
                low = T.bitwise_and(u, T.uint32(_ZERO_WORD_LOW7))
                carried = low + T.uint32(_ZERO_WORD_LOW7)
                merged = T.bitwise_or(
                    T.bitwise_or(carried, u), T.uint32(_ZERO_WORD_LOW7)
                )
                inverted = T.bitwise_xor(merged, T.uint32(0xFFFFFFFF))
                canonical = T.bitwise_and(
                    T.shift_right(inverted, T.uint32(7)), T.uint32(_ZERO_WORD_HIGH)
                )
                c_ub[i] = T.reinterpret(canonical, "int32")

    @lru_cache(maxsize=None)
    def _build_generic_single_block(n_total: int, dtype_str: str, threads: int):
        """One block, one untiled GM->UB->compare->UB->GM pass. Correctness-only --

        serves shapes smaller than ``_GENERIC_TILE`` (the only one this round's test
        surface needs is ``n_total=64``, from ``test_multi_dtype_instance.py``'s
        every-declared-dtype smoke; round_1_schedule.md §6).
        """

        @T.prim_func
        def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), "bool")):
            with T.Kernel(1):
                a = T.alloc_shared((n_total,), dtype_str)
                c = T.alloc_shared((n_total,), "bool")
                T.copy(A, a)
                _cmp_zero_body(a, c, n_total, dtype_str, threads)
                T.copy(c, C)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @lru_cache(maxsize=None)
    def _build_generic_persistent(
        n_total: int, dtype_str: str, num_blocks: int, tile: int, threads: int, iterations: int
    ):
        """Persistent multi-block grid with ``T.Pipelined`` staging.

        Serves every generic-strategy benchmark case (16M-float16, 16M-float32) plus the
        1,048,576-element correctness fixture and the full 9-dtype manifest domain at that
        size (``test_logical.py``/``test_elementwise_compile.py``).
        """

        @T.prim_func
        def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), "bool")):
            with T.Kernel(num_blocks) as bx:
                a = T.alloc_shared((tile,), dtype_str)
                c = T.alloc_shared((tile,), "bool")
                T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
                for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(A[begin : begin + tile], a)
                    _cmp_zero_body(a, c, tile, dtype_str, threads)
                    T.copy(c, C[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_generic_kernel(n_total: int, dtype_str: str):
        """Pick the strategy-1 tiling for ``n_total`` elements of ``dtype_str``."""
        num_blocks, tile, threads, iterations, single_block = _choose_grid(
            n_total, _GENERIC_TILE, _NUM_BLOCKS
        )
        if single_block:
            return _build_generic_single_block(n_total, dtype_str, threads)
        return _build_generic_persistent(n_total, dtype_str, num_blocks, tile, threads, iterations)

    @lru_cache(maxsize=None)
    def _build_pack_single_block(n_packed: int, threads: int):
        """One block, one untiled pass over ``n_packed`` int32 words. Correctness-only --

        serves the bool-pack path's one known tiny shape (16 int32 words, from the
        64-element every-declared-dtype smoke; round_1_schedule.md §6).
        """

        @T.prim_func
        def main(A: T.Tensor((n_packed,), "int32"), C: T.Tensor((n_packed,), "int32")):
            with T.Kernel(1):
                a = T.alloc_shared((n_packed,), "int32")
                c = T.alloc_shared((n_packed,), "int32")
                T.copy(A, a)
                _pack_zero_word_body(a, c, n_packed, threads)
                T.copy(c, C)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @lru_cache(maxsize=None)
    def _build_pack_persistent(
        n_packed: int,
        num_blocks: int,
        tile: int,
        iterations: int,
        num_stages: int = _NUM_STAGES,
    ):
        """Persistent multi-block int32 XOR grid -- this round's primary optimization.

        Serves both bool benchmark cases (16M-bool at 8 iterations, 256M-bool at 128
        iterations, round_1_schedule.md §5.2) and the bool-pack M3 fixture (262,144
        packed elements, 32 blocks, 1 iteration -- see ``_choose_grid``'s docstring).
        """

        @T.prim_func
        def main(A: T.Tensor((n_packed,), "int32"), C: T.Tensor((n_packed,), "int32")):
            with T.Kernel(num_blocks) as bx:
                a = T.alloc_shared((tile,), "int32")
                c = T.alloc_shared((tile,), "int32")
                T.annotate_buffer_versions({a: num_stages, c: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(A[begin : begin + tile], a)
                    _pack_zero_word_body(a, c, tile, threads=min(tile, 2048))
                    T.copy(c, C[begin : begin + tile])
        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_bool_pack_kernel(n_packed: int, num_stages: int = _NUM_STAGES):
        """Pick the strategy-2 (zero-word) tiling for ``n_packed`` int32 words."""
        num_blocks, tile, threads, iterations, single_block = _choose_grid(
            n_packed, _PACK_TILE, _NUM_BLOCKS
        )
        if single_block:
            return _build_pack_single_block(n_packed, threads)
        return _build_pack_persistent(
            n_packed, num_blocks, tile, iterations, num_stages=num_stages
        )

    @T.macro
    def _pack_cmp_zero_body(a_ub, c_ub, n_packed: int, dtype_str: str, threads: int) -> None:
        """``c_ub[:n_packed] = pack4(a_ub[4*i : 4*i+4] == 0)`` -- round 2's strategy 3.

        Each output lane ``i`` reads 4 independent input elements, compares each to zero
        (a genuine per-lane comparison, not a bit-reinterpret -- module docstring's
        Strategy 3 section explains why that is required here, unlike the bool path's
        free XOR trick), and assembles the four 0/1 results into one ``int32`` word with
        lane ``k`` in byte ``k`` (lane 0 unshifted/low byte, lane 3 shifted 24 bits/high
        byte) -- the same little-endian byte layout ``forward()``'s
        ``.view(torch.uint8).view(torch.bool)`` unpack assumes, verified bit-exact on the
        CPU host before this macro was written (round_2_schedule.md).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's
        ``for i in T.Parallel(n):`` is not iterable under TileLang's eager builder
        (confirmed precedent: ``_cmp_zero_body``/``_pack_xor_body`` above).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n_packed):
                b0 = T.cast(a_ub[4 * i + 0] == T.cast(0, dtype_str), "int32")
                b1 = T.cast(a_ub[4 * i + 1] == T.cast(0, dtype_str), "int32")
                b2 = T.cast(a_ub[4 * i + 2] == T.cast(0, dtype_str), "int32")
                b3 = T.cast(a_ub[4 * i + 3] == T.cast(0, dtype_str), "int32")
                lo = T.bitwise_or(b0, T.shift_left(b1, T.cast(8, "int32")))
                hi = T.bitwise_or(
                    T.shift_left(b2, T.cast(16, "int32")), T.shift_left(b3, T.cast(24, "int32"))
                )
                c_ub[i] = T.bitwise_or(lo, hi)

    @lru_cache(maxsize=None)
    def _build_float_pack_single_block(n_total: int, dtype_str: str, n_packed: int, threads: int):
        """One block, one untiled pass. Correctness-only -- serves the strategy-3 path's

        one known tiny shape (``n_total=64`` -> 16 packed int32 words, from the
        64-element every-declared-dtype smoke; round_1_schedule.md §6, same shape
        Strategy 2's ``_build_pack_single_block`` already serves for bool).
        """

        @T.prim_func
        def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_packed,), "int32")):
            with T.Kernel(1):
                a = T.alloc_shared((n_total,), dtype_str)
                c = T.alloc_shared((n_packed,), "int32")
                T.copy(A, a)
                _pack_cmp_zero_body(a, c, n_packed, dtype_str, threads)
                T.copy(c, C)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @lru_cache(maxsize=None)
    def _build_float_pack_persistent(
        n_total: int, dtype_str: str, num_blocks: int, tile_in: int, tile_packed: int, iterations: int
    ):
        """Persistent multi-block grid with ``T.Pipelined`` staging -- round 2's fix for

        the float16/float32 M4 miss (module docstring's Strategy 3 section). Serves both
        float benchmark cases (16M-float16, 16M-float32) plus the 1,048,576-element
        correctness fixture (``test_logical.py``/``test_elementwise_compile.py``) and the
        full float slice of the 9-dtype manifest domain at that size.
        """

        @T.prim_func
        def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total // 4,), "int32")):
            with T.Kernel(num_blocks) as bx:
                a = T.alloc_shared((tile_in,), dtype_str)
                c = T.alloc_shared((tile_packed,), "int32")
                T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
                for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                    begin_in = (step * num_blocks + bx) * tile_in
                    begin_out = (step * num_blocks + bx) * tile_packed
                    T.copy(A[begin_in : begin_in + tile_in], a)
                    _pack_cmp_zero_body(a, c, tile_packed, dtype_str, threads=min(tile_packed, 2048))
                    T.copy(c, C[begin_out : begin_out + tile_packed])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_float_pack_kernel(n_total: int, dtype_str: str):
        """Pick the strategy-3 (float 4-lane pack-compare) tiling for ``n_total`` elements
        of ``dtype_str`` (one of ``_FLOAT_PACK_TILE``'s 3 keys).

        Operates ``_choose_grid`` in *packed* (int32) units -- ``n_packed = n_total // 4``
        against the dtype's packed tile width -- then scales the winning tile back up by
        4x for the actual float input buffer. This is the same grid-sizing function
        Strategy 2 (bool pack) already uses in the same units, just with a dtype-dependent
        tile instead of bool's fixed ``_PACK_TILE``.
        """
        n_packed = n_total // 4
        num_blocks, tile_packed, threads, iterations, single_block = _choose_grid(
            n_packed, _FLOAT_PACK_TILE[dtype_str], _NUM_BLOCKS
        )
        if single_block:
            return _build_float_pack_single_block(n_total, dtype_str, n_packed, threads)
        return _build_float_pack_persistent(
            n_total, dtype_str, num_blocks, tile_packed * 4, tile_packed, iterations
        )


class AscendLogicalNotKernel(Kernel, UnaryPredicateFwdInterface):
    """Ascend implementation of ``LogicalNotFwdOp``: ``y = not x``, bandwidth-bound, no Cube.

    Implements ``UnaryPredicateFwdInterface`` -- **not** ``UnaryElementwiseFwdInterface``,
    whose ``forward(self, input)`` signature looks identical but which this op does not
    declare (``LogicalNotFwdOp.interfaces = {ELEMENTWISE: UnaryPredicateFwdInterface}``,
    ``tileops/ops/elementwise/logical.py``). Getting this wrong would leave this class
    implementing none of the op's declared interfaces, failing construction on every
    device as soon as this module is imported (round_1_schedule.md §2, citing the
    ``ReciprocalFwdOp`` round-1 precedent for the same mistake).

    Covers the manifest's full 9-dtype domain (bool, uint8, int8, int16, int32, int64,
    float16, bfloat16, float32) -- ``refusal()`` is left at ``Kernel``'s default (serves
    every call), matching the CUDA side's ``LogicalUnaryKernel.SUPPORTED_DTYPES ==
    LOGICAL_DTYPES`` (all 9, no device-specific int fallback to be gapped around; module
    docstring, round_1_schedule.md §2). Internally dispatches (element count permitting)
    bool to the 4-pack XOR strategy and float16/bfloat16/float32 to the round-2 4-lane
    pack-compare strategy; every other dtype (and bool/float when the element count is
    not a multiple of 4, which no known case hits) falls back to the generic
    compare-to-zero strategy.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    # supported_archs left at Kernel's None default: no Ascend SoC-version gate exists to
    # replace CUDA's SM-arch one (Kernel._check_arch skips the check when None).

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if T is None:
            raise ImportError(
                "AscendLogicalNotKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        dtype_str = self.dtype_to_str(dtype)
        # Strategy 2/3 both need n_total % 4 == 0 for their int32-packing view chain;
        # every manifest workload and test fixture this round knows about satisfies it
        # (64, 1_048_576, 2**24, 2**28 are all powers of two >= 4). Falls back to
        # strategy 1 (still correct, just without the 4x packing win) if some future
        # caller does not.
        divisible_by_4 = n_total % 4 == 0
        self._use_bool_pack = dtype == torch.bool and divisible_by_4
        self._use_float_pack = dtype_str in _FLOAT_PACK_TILE and divisible_by_4
        if self._use_bool_pack:
            self.kernel = _build_bool_pack_kernel(n_total // 4)
        elif self._use_float_pack:
            self.kernel = _build_float_pack_kernel(n_total, dtype_str)
        else:
            self.kernel = _build_generic_kernel(n_total, dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``torch.logical_not(input)``; nothing is written in place.

        Args:
            input: ``self.n_total`` elements on an NPU device, contiguous in whatever
                shape the caller passed, in ``self.dtype``.

        Returns:
            A new contiguous ``torch.bool`` tensor on the same device, shaped as
            ``input``.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on "
                f"{input.device}. Another target's backend serves other devices."
            )
        shape = input.shape
        flat_in = input.reshape(-1)
        if flat_in.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_in.numel()} elements"
            )
        if self._use_bool_pack:
            packed_in = flat_in.view(torch.uint8).view(torch.int32)
            packed_out = self.kernel(packed_in)
            result = packed_out.view(torch.uint8).view(torch.bool)
        elif self._use_float_pack:
            # No input-side reinterpret here (unlike strategy 2): the kernel reads
            # `flat_in` in its native float dtype and does 4 genuine per-lane compares
            # (module docstring's Strategy 3 section) -- only the *output* is a packed
            # int32 buffer that needs unpacking back to bool on the way out.
            packed_out = self.kernel(flat_in)
            result = packed_out.view(torch.uint8).view(torch.bool)
        else:
            result = self.kernel(flat_in)
        return result.reshape(shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("LogicalNotFwdOp", "ascend", AscendLogicalNotKernel)
except ImportError:
    pass
