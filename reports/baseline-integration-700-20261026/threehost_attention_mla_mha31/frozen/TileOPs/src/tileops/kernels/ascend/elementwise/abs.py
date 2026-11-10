"""Ascend abs kernel: ``y = |x|``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``AbsFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA keys ``"abs"``
(``tileops.kernels.elementwise.math_unary.AbsFwdKernel``) and ``"abs_int"``
(``AbsIntFwdKernel``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``AbsFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/AbsFwdOp/round_1_schedule.md`` (Planner, round 1, corrected once
-- see its §3). ``|x|`` is memory-bound and exact in every finite precision (it only clears
the sign bit) -- no fp32 promotion, mirroring ``AbsFwdKernel.op_func`` (``return T.abs(x)``)
exactly in spirit, though not in literal form (see below). Tiling directly reuses
``ReluFwdOp``'s round-2-calibrated persistent-grid configuration verbatim: ``num_blocks=64``
(live-queried via ``get_num_vector_cores``, floored to the nearest power of two, falling back
to 64), ``tile=8192``, ``num_stages=2``, ``threads=min(2048, tile)=2048``. Both of this op's
manifest shapes (``2**24``, ``2**28``) divide ``num_blocks * tile = 2**19`` with zero remainder
(32 and 512 pipelined iterations respectively), so round 1 ships exactly the same tiling shape
``relu.py``/``sqrt.py`` already validated for this SKU.

**Round 2** (``docs/ascend950/dev/AbsFwdOp/round_2_schedule.md``) swept ``tile`` per dtype
across ``{8192, 4096, 2048}`` -- more ``T.Pipelined`` iterations at fixed ``num_blocks=64``/
``num_stages=2`` -- to test whether round 1's 16M/float16/bfloat16 misses were launch/pipeline-
fill-drain overhead rather than bandwidth, per the Planner's arithmetic-only diagnosis (round 1's
own measured latency is nearly dtype-flat at 16M, so the fp16/bf16-vs-fp32 ratio gap is
baseline-side, not kernel-side -- see the schedule's §1). The lever did not help: both smaller
tile values regressed **every one of the 5 benchmarked cases, monotonically with smaller tile,
on all 3 dtypes including float32** (round 1's one passing case) -- the opposite of the
predicted direction, not a null result. ``tile=8192`` is unchanged on all 3 dtypes; see
``_TILE_BY_DTYPE`` below and ``round_2_result.md`` for the full sweep table and the
backlog recommendation.

**``T.abs`` dtype-generic dispatch -- confirmed NOT usable at fp16/bf16 on this toolchain, by
running the minimal repro the schedule asked for, not by assumption.** The schedule flagged
the 16-bit (``hfabs``) path as the one residual risk (fp32's ``fabsf`` path is proven by
``example_per_token_cast_to_fp8.py``) and asked for one cheap standalone npu-guard pre-check
before wiring the full kernel. That pre-check (a tiny 1-block, ``n=128`` kernel, fp16/bf16/fp32)
found the risk is real, and worse than "hfabs doesn't resolve" alone -- it fails in two
*different* ways depending on dtype:

- **float16**: compiles down to the ``.asc`` C++ source and only then fails at the ``bisheng``
  backend-compile step: ``error: use of undeclared identifier 'hfabs'`` -- the
  ``AscendMath``/``REGISTER_ASCEND_INTRIN("fabs")`` dispatch (``intrin_rule_ascend.cc``) emits a
  call to a symbol the toolchain's vector-intrinsic headers do not actually declare.
- **bfloat16**: fails *earlier*, at TIR lowering, before any C++ is even generated:
  ``InternalError: Unresolved call ir.Op(... name="tirx.fabs" ...)`` -- the generic
  ``tirx.fabs`` op itself has no backend lowering registered for this storage type on this
  build, a different failure stage than fp16's "wrong symbol name at codegen".

Per the schedule's zero-new-risk fallback (same shape as ``relu.py``'s ``_relu_body``, just
without the ``max``), ``_abs_body`` below uses ``T.if_then_else(x >= 0, x, -x)`` for **all
three** float dtypes, not just fp16/bf16: this also matches ``tvm::abs``'s own generic
(no-native-intrinsic) lowering (``Select(x >= 0, x, -x)``), so it is not a guess, and using one
uniform code path for every dtype (rather than branching float32 to ``T.abs`` and fp16/bf16 to
the comparison form) keeps the kernel body simple with no extra risk -- fp32's compare-and-negate
is at least as cheap as a ``fabsf`` call and this op is memory-bound regardless. ``-x_ub[i]``
itself is a already-proven path (``NegFwdOp``'s ``_neg_body`` compiles and runs bare ``-x_ub[i]``
correctly at all 3 float dtypes).

**Integer dtypes -- a correctness requirement, not a performance one** (schedule doc §3,
corrected): the manifest's ``forall`` also admits 5 integer dtypes (int8/int16/int32/int64/
uint8), served on CUDA by ``AbsIntFwdKernel(TorchFallbackKernel)``, which does not override
``Kernel.devices`` (stays ``frozenset({"cuda"})``) and is therefore ``unavailable()`` on an
``npu`` call. If ``AscendAbsKernel.refusal()`` only accepted float dtypes, an ``npu`` + int call
would find zero eligible implementations and ``key_for`` would raise ``ValueError: no
implementation serves this call``. ``refusal()`` therefore accepts the full manifest domain
(the 3 floats plus ``INT_DTYPES``, imported from ``tileops.kernels.elementwise.math_unary`` and
not redefined here), and ``forward()`` branches internally: floats run the TileLang program
below; ints are answered via ``torch.abs``, matching ``AbsIntFwdKernel.handler`` bit-for-bit.

**Checked against a live sibling-op finding before assuming this was safe**: ``NegFwdOp``'s
worker found that this NPU build's ``aclnnNeg`` does not implement int16/uint8 (an
``AclNN_Parameter_Error`` at the ACL layer), even though CUDA's ``torch.neg`` handles every int
width the manifest admits -- a real ACLNN operator-coverage gap, not a dispatch issue. The same
pre-check run for this op (standalone script, ``torch.abs`` called directly on NPU-resident
int8/int16/int32/int64/uint8 tensors, each cross-checked against a CPU-computed
``torch.abs`` reference) found **``aclnnAbs`` does NOT have the same gap**: all 5 int dtypes
answer correctly and bit-exactly via a plain, unmodified ``torch.abs(input)`` call --

```
torch.int8:  torch.abs direct OK exact_match=True
torch.int16: torch.abs direct OK exact_match=True
torch.int32: torch.abs direct OK exact_match=True
torch.int64: torch.abs direct OK exact_match=True
torch.uint8: torch.abs direct OK exact_match=True
```

So, unlike ``AscendNegKernel``, this kernel's int branch needs no int32 promote/truncate
round-trip -- ``forward()``'s int path is exactly the one-line passthrough the (corrected)
schedule originally proposed, now confirmed rather than assumed. (See
``docs/ascend950/dev/AbsFwdOp/round_1_result.md`` §2 for the full pre-check transcript and the
separate, in-process confirmation against ``AscendAbsKernel`` itself via ``AbsFwdOp``.)

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call
are all wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md``
§1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``docs/ascend950/dev_process.md`` §1, several ops' workers write into this same
``kernels/ascend``/``kernels/ascend/elementwise`` directory concurrently, each registering from
its own file, and a shared ``__init__.py`` would be a file they all contend for.
``kernels/ascend`` and ``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420)
namespace packages, matching every other Ascend elementwise kernel already in this directory.
"""

import functools

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    import tileops.backend
    from tileops.kernels.elementwise.call_spec import (
        ElementwiseCall,
        UnaryElementwiseFwdInterface,
    )
    from tileops.kernels.elementwise.math_unary import INT_DTYPES
    from tileops.kernels.kernel_base import Entry, Kernel

    # This round's 3 benchmarked float dtypes. INT_DTYPES (the other 5 dtypes the manifest
    # admits for this op) is imported, not redefined, from math_unary.py -- it is the same
    # tuple AbsIntFwdKernel.SUPPORTED_DTYPES already uses on the CUDA side (schedule doc §3).
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

        Reuses ``relu.py``'s round-2-calibrated helper verbatim (schedule doc §4): queries
        this SKU's real vector-core count (``get_num_vector_cores``, 72 on this SKU) and
        floors to the nearest power of two (64), falling back to 64 if the live query raises
        or returns a falsy value at build time. ``ReluFwdOp`` round 2's official (no-cache-
        flush) sweep already found 64/8192/2-stages the outright winner for this exact
        bandwidth-bound unary shape on this SKU -- not re-swept here (schedule doc §4).

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

    # Stage count for the memory-bound, large-n persistent-grid path: num_blocks itself is
    # resolved lazily by _calibrated_num_blocks(). Already independently confirmed
    # best-of-swept (ReluFwdOp round 2) and not re-litigated by round 2's tile sweep
    # (round_2_schedule.md §2).
    _NUM_STAGES = 2

    # Per-dtype tile for the persistent-grid path (round_2_schedule.md §2): round 1 shipped
    # tile=8192 uniformly for all 3 float dtypes. Round 2 swept tile in {4096, 2048} -- more
    # T.Pipelined iterations at fixed num_blocks=64/num_stages=2 -- under the official M4
    # protocol, identical candidate set to NegFwdOp round 2 (same shapes/byte-widths/UB math).
    # Result: both candidates regressed all 5 cases, monotonically with smaller tile, on every
    # dtype including float32 (round_1_result.md's one passing case) -- e.g. 16M-float32 went
    # 0.021 ms (tile=8192) -> 0.028 ms (tile=4096, +33%) -> 0.035 ms (tile=2048, +65%); the
    # 16M fp16/bf16 and 256M cases showed the same monotonic-worse shape. No candidate beat
    # round 1 on any case, so all three dtypes keep tile=8192 unchanged. See round_2_result.md
    # for the full sweep table.
    _TILE_BY_DTYPE = {
        "float16": 8192,
        "bfloat16": 8192,
        "float32": 8192,
    }
    _DEFAULT_TILE = 8192

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against (schedule
    # doc §1: "no decode case" in this op's own manifest workloads). This threshold still
    # matters for two non-benchmarked small shapes this round's required tests use:
    # test_multi_dtype_instance's n_total=64 sweep and (defensively) any future small-n caller.
    # Fixed at round 1's tile value (not tied to the now-per-dtype _TILE_BY_DTYPE): every
    # small-n fixture this round's tests exercise (n_total=64) is far below any candidate tile,
    # so this threshold's exact value does not interact with the round-2 sweep.
    _SINGLE_BLOCK_THRESHOLD = 8192

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _abs_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = |x_ub[:n]|``, compiler-vectorized over ``threads`` lanes.

        Uses the schedule's zero-new-risk fallback form (``T.if_then_else(x >= 0, x, -x)``),
        not ``T.abs``/``T.if_then_else``'s native-intrinsic cousin: a standalone pre-check
        (module docstring) found ``T.abs`` fails to compile at both fp16 (``hfabs`` undeclared
        at the ``bisheng`` backend-compile step) and bf16 (``tirx.fabs`` has no backend
        lowering registered at all for this storage type, failing earlier at TIR lowering) on
        this toolchain. This form matches ``tvm::abs``'s own generic (no-native-intrinsic)
        lowering (``Select(x >= 0, x, -x)``) and is used uniformly for all three float dtypes,
        including float32 (whose native ``fabsf`` path for ``T.abs`` *does* compile) -- one
        code path is simpler than branching per dtype, and compare-and-negate is no more
        expensive than a ``fabsf`` call for a memory-bound op.  ``T.cast(0, x_ub.dtype)``, not a
        bare literal, mirrors ``relu.py``'s ``_relu_body`` so the comparison does not depend on
        literal-to-narrow-dtype promotion behavior. ``-x_ub[i]`` is an already-proven path:
        ``NegFwdOp``'s ``_neg_body`` compiles and runs bare negation correctly at all 3 float
        dtypes.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): TileLang's eager builder rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into its trace-builder calls, so an undecorated
        helper's ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable``.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero = T.cast(0, x_ub.dtype)
                y_ub[i] = T.if_then_else(x_ub[i] >= zero, x_ub[i], -x_ub[i])

    @functools.lru_cache(maxsize=64)
    def _make_abs_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first path for small ``n_total``
        (below ``_SINGLE_BLOCK_THRESHOLD``) -- this op's manifest has no decode-sized
        benchmark case, but ``test_multi_dtype_instance``'s cross-op smoke sweep still drives
        every float dtype at ``n_total=64``, which needs a working small-``n`` path.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _abs_body(x_ub, y_ub, n_total, threads=min(2048, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_abs_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §4).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_abs_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_abs_generic_single_block`` instead. All 5 of this op's
        benchmarked cases, plus the correctness fixture's ``n_total=1_048_576``, hit this path.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_abs_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
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
                    _abs_body(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_abs_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide (schedule doc §4: "write it anyway for defensiveness/future shapes"). One
        block; a main loop over full tile-sized chunks, plus one smaller, statically-sized tail
        chunk. Not performance-tuned: none of this op's manifest cases or required test
        fixtures reach this function this round (every shape this round exercises -- 64,
        1_048_576, 2**24, 2**28 -- either fits the single-block path or divides
        ``num_blocks * tile`` exactly) -- written for defensiveness and for the interface's
        general contract (an arbitrary shape, not just the benchmarked ones), mirroring
        ``relu.py``'s/``sqrt.py``'s ``_make_*_generic_single_block``.
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
                        _abs_body(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _abs_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_abs_kernel(n_total: int, dtype: str, tile: int):
        """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback.

        ``tile`` is a parameter, not a hardcoded local (code-style.md: a value only one
        function reads is a local) -- round 2 (``round_2_schedule.md`` §2) sweeps it per
        dtype across ``{8192, 4096, 2048}`` before ``AscendAbsKernel.__init__`` settles on
        one value per dtype to ship, via ``_TILE_BY_DTYPE`` (``round_2_result.md``).
        """
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            return _make_abs_single_block(n_total, dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_abs_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_abs_generic_single_block(n_total, dtype, tile)

    class AscendAbsKernel(Kernel, UnaryElementwiseFwdInterface):
        """Absolute value on Ascend NPUs: ``y = |x|``.

        Memory-bound (one load, one compare-and-conditional-negate, one store per element);
        exact at every finite value regardless of storage width, since clearing a sign bit
        loses nothing (``docs/ascend950/dev/AbsFwdOp/round_1_schedule.md`` §2). A fresh class
        per ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the
        CUDA dialect facade).

        Accepts the op's *full* manifest domain -- the 3 benchmarked float dtypes plus the 5
        integer dtypes the manifest also admits -- not just the floats (schedule doc §3):
        ``forward()`` runs the TileLang program below for float input, and for integer input
        reproduces ``AbsIntFwdKernel.handler``'s result bit-for-bit via a direct ``torch.abs``
        call (confirmed, not assumed, to have no NPU-side dtype coverage gap -- see the module
        docstring), so an ``npu`` + int call still has an eligible implementation once this
        module is imported.
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: ElementwiseCall) -> "str | None":
            if call.dtype not in _FLOAT_DTYPES and call.dtype not in INT_DTYPES:
                return f"serves {_FLOAT_DTYPES + INT_DTYPES}, not {call.dtype}"
            return super().refusal(call)

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._is_int = dtype in INT_DTYPES
            if self._is_int:
                self._compiled_fn = None
            else:
                dtype_str = _dtype_str(dtype)
                tile = _TILE_BY_DTYPE.get(dtype_str, _DEFAULT_TILE)
                self._compiled_fn = _make_abs_kernel(n_total, dtype_str, tile)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                # Bit-identical to AbsIntFwdKernel.handler (math_unary.py:270) -- torch.abs
                # already answers every one of the 5 int dtypes correctly on an npu tensor,
                # with no promote/truncate round-trip needed (unlike AscendNegKernel's
                # int16/uint8 path): confirmed live, not assumed (module docstring).
                # UnaryOp.forward() already made `input` contiguous before this call; the
                # extra .contiguous() is cheap defensive parity with
                # TorchFallbackKernel.forward()'s own explicit call.
                return torch.abs(input).contiguous()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("AbsFwdOp", "ascend", AscendAbsKernel)

except ImportError:
    pass
