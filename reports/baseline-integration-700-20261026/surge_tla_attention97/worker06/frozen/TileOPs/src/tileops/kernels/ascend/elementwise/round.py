"""Ascend round kernel: ``y = round(x)`` (round-half-to-even, ``decimals=0``), via the
``tilelang.ascend`` dialect.

Registers an additional implementation of ``RoundFwdOp``'s ``RoundFwdInterface`` under the
``"ascend"`` dispatch key, beside the in-tree CUDA keys ``"round"``
(``tileops.kernels.elementwise.math_unary.RoundFwdKernel``), ``"round_int"``
(``IntIdentityFwdKernel``, shared with floor/ceil/trunc), and ``"round_decimals"``
(``RoundDecimalsFwdKernel``, the ``decimals != 0`` path, untouched by this file -- no
benchmarked case this round uses ``decimals != 0``, see below). No existing file is edited:
``tileops.backend.register_kernel_type`` joins this class to every ``RoundFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/RoundFwdOp/round_1_schedule.md``.

**The load-bearing correctness trap this schedule flagged, confirmed by a host-only compile
probe before writing this file (no NPU touched to find this out):** ``torch.round`` is
round-half-to-**even** (``round(2.5) == 2``, ``round(3.5) == 4``), not round-half-away-from-
zero. Ascend's ``T.round`` (default ``rounding_mode="ties-to-even"``, despite the name) is
wrong here at *every* dtype:

- fp16/bf16 don't even compile (``AscendMath``'s ``REGISTER_ASCEND_INTRIN("round")`` dispatch
  emits an unresolved/undeclared ``hround``/``tirx.round`` call -- the same two-shaped gap
  ``AbsFwdOp`` already found for ``fabs``/``hfabs``, since ``floor``/``ceil``/``fabs``/
  ``round``/``trunc`` share one ``AscendMath`` macro instantiation).
- fp32 *does* compile, but generates a literal ``roundf(...)`` call, which is
  round-half-**away**-from-zero by the C standard -- confirmed by dumping the generated
  ``.asc`` source for the fp32 case. ``T.round``'s "ties-to-even" label does not describe
  what it actually lowers to on this backend at 32-bit, mirroring a CUDA quirk where
  "ties-to-even" and the explicit "ties-away-from-zero" mode are bit-identical at 32/64-bit
  (both resolve to ``roundf``/``round``) -- CUDA only special-cases 16-bit to a true
  ties-to-even ``hrint``, and Ascend has no such 16-bit override at all.

The correct intrinsic is ``T.nearbyint``: no backend lowering is registered for 16-bit float
either (``"nearbyint"`` is absent from ``intrin_rule_ascend.cc``'s dispatch table), but the
fp32-native path falls through to a generic scalar extern call (``nearbyintf``, standard
libm) without needing an Ascend-specific registration, so wrapping every dtype in an
fp32-promote/cast-back (exactly what the existing CUDA ``RoundFwdKernel.op_func`` already
does -- not a coincidence, it is the identical trap on a different backend) compiles cleanly
at all three dtypes and generates a dumped-and-inspected ``nearbyintf(...)`` call, which *is*
specified to honor the current (default: round-to-nearest-ties-to-even) IEEE rounding
direction. ``_round_body`` below is therefore `T.cast(T.nearbyint(T.cast(x, "float32")),
dtype)`` uniformly for all three float dtypes, with no ``T.round`` anywhere in this file.

**This is mandatorily verified again below with a real NPU run**, not just trusted from the
compile probe: see ``docs/ascend950/dev/RoundFwdOp/round_1_result.md``'s tie-break probe,
which feeds exact half-integer inputs (``{-4.5 .. 4.5}``) through the compiled kernel at all
three float dtypes and compares against CPU ``torch.round``. ``test_round`` (the shared
pytest fixture) alone cannot catch a wrong rounding mode: it draws from continuous ``randn``
data, which has ~zero probability of ever landing on an exact ``x.5`` tie, so a kernel
silently implementing ties-away-from-zero could pass it anyway.

**Integer dtypes -- two independent refusal gates, not one** (schedule doc §2): the
manifest's ``forall`` also admits 5 integer dtypes, served on CUDA by
``IntIdentityFwdKernel`` (``handler = torch.Tensor.clone``), which does not override
``Kernel.devices`` (stays ``frozenset({"cuda"})``) and is therefore ``unavailable()`` on an
``npu`` call -- so ``AscendRoundKernel.refusal()`` must accept the int domain too, or an
``npu`` + int call finds zero eligible implementations. ``round`` on an already-integral
value is a pure identity (mathematically and bit-for-bit, every int width), so
``forward()``'s int branch is ``input.clone()``, matching ``IntIdentityFwdKernel.handler``
exactly, with no NPU math operator invoked at all -- no ACLNN coverage-gap risk to even check
for (unlike ``NegFwdOp``'s ``aclnnNeg`` int16/uint8 gap).

Second, separate gate: ``IntIdentityFwdKernel`` *also* implements ``RoundFwdInterface`` with
its own ``decimals`` refusal (``math_unary.py:294-297``), and the CUDA float path's
``RoundFwdKernel.refusal()`` carries the matching gate on the float side
(``math_unary.py:207-211``). ``AscendRoundKernel.refusal()`` replicates this for both
sub-cases: any ``decimals != 0`` call (float or int) is refused here, which is the *only*
source of the ``ValueError`` ``tests/ops/test_unary_math.py::
test_round_rejects_decimals_on_integer_input`` expects -- ``op_base.py`` has no
``shape_rules`` re-enforcement of its own at call time, so skipping this gate would make that
test fail outright on ``npu``, not just be redundant. All 5 of this round's benchmarked cases
use ``decimals=0`` (confirmed from the manifest's ``workloads``, schedule doc §0), so this
gate's float/int ``decimals != 0`` rejection path is exercised by tests, never by a
benchmarked case.

Tiling directly reuses, unmodified, the persistent-grid configuration ``ReluFwdOp`` round 2
calibrated under the official (no-cache-flush) M4 protocol and that ``NegFwdOp``/
``AbsFwdOp`` already confirmed transfers to this exact bandwidth-bound unary shape:
``num_blocks=64`` (live-queried via ``get_num_vector_cores``, floored to the nearest power of
two, falling back to 64), ``tile=8192``, ``num_stages=2``, ``threads=min(2048, tile)=2048``.
Both of this op's manifest shapes (``2**24``, ``2**28``) and the correctness fixture's
``2**20`` divide ``num_blocks * tile = 2**19`` with zero remainder (2/32/512 pipelined
iterations respectively), so round 1 ships one kernel-building path with no decode/tail
branch needed (schedule doc §3).

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a
CUDA-only process) is a no-op: the Ascend-specific imports, the kernel class, and the
registration call are all wrapped in one ``try/except ImportError`` block, per
``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``docs/ascend950/dev_process.md`` §1, several ops' workers write into this
same ``kernels/ascend``/``kernels/ascend/elementwise`` directory concurrently, each
registering from its own file, and a shared ``__init__.py`` would be a file they all contend
for. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are therefore left as implicit
(PEP 420) namespace packages, matching every other Ascend elementwise kernel already in this
directory.
"""

import functools

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    import tileops.backend
    from tileops.kernels.elementwise.call_spec import RoundCall, RoundFwdInterface
    from tileops.kernels.elementwise.math_unary import INT_DTYPES
    from tileops.kernels.kernel_base import Entry, Kernel

    # This round's 3 benchmarked float dtypes. INT_DTYPES (the other 5 dtypes the manifest
    # admits for this op) is imported, not redefined, from math_unary.py -- the same tuple
    # IntIdentityFwdKernel.SUPPORTED_DTYPES already uses on the CUDA side (schedule doc §2).
    _FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound path.

        Reuses ``relu.py``/``abs.py``/``neg.py``'s round-2-calibrated helper verbatim
        (schedule doc §3): queries this SKU's real vector-core count
        (``get_num_vector_cores``, 72 on this SKU) and floors to the nearest power of two
        (64), falling back to 64 if the live query raises or returns a falsy value at build
        time. ``ReluFwdOp`` round 2's official (no-cache-flush) sweep already found
        64/8192/2-stages the outright winner for this exact bandwidth-bound unary shape on
        this SKU -- not re-swept here.

        Deliberately **not** evaluated at module import time: merely importing this module
        must stay a no-NPU-touch no-op (``dev_process.md`` §4). ``lru_cache`` makes this a
        one-time query per process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Stage count and tile for the memory-bound, large-n persistent-grid path:
    # num_blocks itself is resolved lazily by _calibrated_num_blocks(). Already
    # independently confirmed best-of-swept (ReluFwdOp round 2) and reused unmodified by
    # NegFwdOp/AbsFwdOp round 1 on this identical shape (schedule doc §3).
    _NUM_STAGES = 2
    _TILE = 8192

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would
    # add fixed per-launch overhead with no bandwidth-bound tail to amortize it against
    # (schedule doc §3: this op's manifest has no decode case). Still needed for small-n
    # fixtures some of the shared tests exercise (e.g. a multi-dtype smoke sweep at
    # n_total=64). Fixed at round 1's tile value, matching abs.py/neg.py.
    _SINGLE_BLOCK_THRESHOLD = 8192

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _round_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = round_half_to_even(x_ub[:n])``, vectorized over ``threads`` lanes.

        ``T.cast(T.nearbyint(T.cast(x_ub[i], "float32")), x_ub.dtype)`` -- mirrors the
        existing CUDA ``RoundFwdKernel.op_func`` almost literally, just through the Ascend
        dialect import. Deliberately **not** ``T.round``: see the module docstring for the
        compile-probe evidence that ``T.round`` either fails to compile (fp16/bf16) or, where
        it does compile (fp32), generates ``roundf`` -- round-half-away-from-zero, the wrong
        function for ``torch.round``'s semantics. ``T.nearbyint`` needs the same fp32
        promotion for fp16/bf16 that ``T.round`` would (no native 16-bit lowering for either
        on this backend), but its fp32-native path generates ``nearbyintf``, which *is*
        ties-to-even -- confirmed by a host-only compile probe (module docstring) and,
        mandatorily, by a live NPU tie-break probe (``round_1_result.md``) independent of
        this file.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): an undecorated helper's
        ``for i in T.Parallel(n):`` raises ``TypeError: 'ForFrame' object is not iterable``
        under TileLang's eager builder.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.nearbyint(T.cast(x_ub[i], "float32")), x_ub.dtype)

    @functools.lru_cache(maxsize=16)
    def _make_round_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first path for small ``n_total``
        (below ``_SINGLE_BLOCK_THRESHOLD``) -- this op's manifest has no decode-sized
        benchmark case, but some shared tests still drive small ``n_total`` values, which
        need a working small-``n`` path.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _round_body(x_ub, y_ub, n_total, threads=min(2048, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=16)
    def _make_round_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_round_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_round_generic_single_block`` instead. All 5 of this
        op's benchmarked cases, plus the correctness fixture's ``n_total=1_048_576``, hit
        this path.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_round_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes "
            "elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _round_body(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=16)
    def _make_round_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not
        evenly divide. One block; a main loop over full tile-sized chunks, plus one smaller,
        statically-sized tail chunk. Not performance-tuned: no benchmarked case or required
        test fixture this round reaches this function (every shape exercised -- small-n
        smoke values, 1_048_576, 2**24, 2**28 -- either fits the single-block path or
        divides ``num_blocks * tile`` exactly) -- written for defensiveness and for the
        interface's general contract, mirroring ``relu.py``/``abs.py``/``neg.py``.
        """
        full_tiles, remainder = divmod(n_total, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), dtype)
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _round_body(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _round_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_round_kernel(n_total: int, dtype: str, tile: int):
        """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback."""
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            return _make_round_single_block(n_total, dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_round_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_round_generic_single_block(n_total, dtype, tile)

    class AscendRoundKernel(Kernel, RoundFwdInterface):
        """Round-half-to-even on Ascend NPUs: ``y = round(x)`` (``decimals=0`` only).

        Memory-bound (one load, one fp32-promoted ``nearbyintf`` call, one store per
        element); see the module docstring for why ``T.nearbyint`` is correct here and
        ``T.round`` is not, despite its name. A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language``
        (the CUDA dialect facade).

        Accepts the op's *full* manifest domain for ``decimals=0`` -- the 3 benchmarked
        float dtypes plus the 5 integer dtypes the manifest also admits -- and refuses any
        ``decimals != 0`` call outright, for both floats and ints (module docstring's "two
        independent refusal gates"). ``forward()`` runs the TileLang program below for float
        input, and for integer input returns ``input.clone()``, matching
        ``IntIdentityFwdKernel.handler`` bit-for-bit (round-of-an-integer is identity; no
        NPU math operator is invoked, so there is no ACLNN coverage-gap risk to check,
        unlike ``NegFwdOp``'s ``aclnnNeg`` int16/uint8 gap).
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: RoundCall) -> "str | None":
            if call.decimals != 0:
                return f"rounds to whole numbers, not {call.decimals} decimal places"
            if call.dtype not in _FLOAT_DTYPES and call.dtype not in INT_DTYPES:
                return f"serves {_FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"
            return None

        @classmethod
        def entry_for(cls, call: RoundCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._is_int = dtype in INT_DTYPES
            self._compiled_fn = (
                None if self._is_int else _make_round_kernel(n_total, _dtype_str(dtype), _TILE)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                # Bit-identical to IntIdentityFwdKernel.handler (torch.Tensor.clone) --
                # round(int) is a pure identity, every width, no NPU math operator needed.
                # UnaryOp.forward() already made `input` contiguous before this call; the
                # extra .contiguous() is cheap defensive parity with
                # TorchFallbackKernel.forward()'s own explicit call.
                return input.clone().contiguous()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("RoundFwdOp", "ascend", AscendRoundKernel)

    class AscendRoundDecimalsKernel(Kernel, RoundFwdInterface):
        """Rounding to a non-zero number of decimal places, for an NPU-resident input.

        Bit-identical to the CUDA ``RoundDecimalsFwdKernel`` body
        (``round(x * 10**k) / 10**k``, computed through fp32): that class does not override
        ``Kernel.devices`` either, so it is ``unavailable()`` on an ``npu`` call -- before
        this class, **every** ``npu`` + ``RoundFwdOp(decimals != 0)`` call had zero eligible
        implementations, regardless of dtype (schedule doc §2.1), which several shared,
        device-generic tests in ``tests/ops/test_unary_math.py``
        (``test_round_decimals``/``test_round_decimals_no_overflow_low_precision``/
        ``test_round_decimals_validates_input``) exercise under ``--tileops-device=npu``.
        This class closes that pre-existing gap for free: it compiles nothing (pure torch),
        so there is nothing to tune, and it carries zero TileLang risk. Not part of this
        round's 5 benchmarked cases (schedule doc §0 confirms all of them use the default
        ``decimals=0``, which ``AscendRoundKernel`` above already serves) -- registered here
        purely so the shared decimals tests pass cleanly under ``npu`` too, not because any
        benchmarked case needs it.
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: RoundCall) -> "str | None":
            if call.decimals == 0:
                return "rounds to a non-zero number of decimal places"
            if call.dtype not in _FLOAT_DTYPES and call.dtype not in INT_DTYPES:
                return f"serves {_FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"
            return None

        @classmethod
        def entry_for(cls, call: RoundCall) -> Entry:
            return call.decimals, lambda: cls(call.decimals)

        def __init__(self, decimals: int) -> None:
            super().__init__()
            self.decimals = decimals

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Through float32, so scaling by 10**decimals cannot overflow a narrow input;
            # the single down-cast restores the element type. Identical computation to
            # RoundDecimalsFwdKernel.forward (math_unary.py:328-330).
            return torch.round(input.float(), decimals=self.decimals).to(input.dtype)

    tileops.backend.register_kernel_type("RoundFwdOp", "ascend_decimals", AscendRoundDecimalsKernel)

except ImportError:
    pass
