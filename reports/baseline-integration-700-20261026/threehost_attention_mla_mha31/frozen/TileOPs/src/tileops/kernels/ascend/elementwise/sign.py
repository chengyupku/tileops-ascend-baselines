"""Ascend sign kernel: ``y = sign(x)`` (-1, 0, or +1), via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``SignFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA keys ``"sign"``
(``tileops.kernels.elementwise.math_unary.SignFwdKernel``) and ``"sign_int"``
(``SignIntFwdKernel``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``SignFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/SignFwdOp/round_1_schedule.md``. ``sign(x)`` is memory-bound
(one load, a 2-compare/select chain, one store per element). ``tilelang.ascend.language`` has
no ``sign`` intrinsic (``hasattr(T, "sign") == False``; the only ``*sign*``-named export is
``copysign``, a different function that does not reproduce ``torch.sign(0) == 0``), so this
ships a 3-way ``T.if_then_else`` comparison chain -- the same logical structure
``SignFwdKernel.op_func`` already uses on the CUDA side -- **without** that kernel's fp32
promotion: a comparison against a dtype-native zero and a dtype-native ``{-1,0,1}`` select is
exact at every finite fp16/bf16/fp32 value (no accumulation, no narrow-dtype overflow risk),
the same reasoning ``NegFwdKernel``'s bare ``-x`` and ``AbsFwdKernel``'s bare ``T.abs(x)``
already rely on, and the same reasoning ``AscendNegKernel``/``AscendAbsKernel`` already ship in
production (schedule doc §2).

Tiling directly reuses ``ReluFwdOp``'s round-2-calibrated persistent-grid configuration
verbatim, the same config ``NegFwdOp``/``AbsFwdOp`` round 1 shipped and whose own round-2 tile
re-sweeps ({4096, 2048}) regressed every case: ``num_blocks=64`` (live-queried via
``get_num_vector_cores``, floored to the nearest power of two, falling back to 64), ``tile=8192``,
``num_stages=2``, ``threads=min(2048, tile)=2048``. Both of this op's manifest shapes (``2**24``,
``2**28``) divide ``num_blocks * tile = 2**19`` with zero remainder (32 and 512 pipelined
iterations respectively), so round 1 ships exactly this one persistent-grid path, plus a
single-block fallback for small ``n_total`` (``test_sign_edge``'s ``n_total=4096``, below the
``tile=8192`` threshold) and a generic tail fallback for defensiveness (schedule doc §4/§5).

**NaN through the comparison chain**: ``test_sign_edge`` drives ``float("nan")`` through this
op's body -- the first Ascend elementwise kernel in this project whose correctness test
actually exercises NaN through a comparison op end-to-end (schedule doc §2). Confirmed on CPU
before assuming it (``torch.sign(torch.tensor(float("nan"))) == 0.0``, not NaN), so the chain's
``zero`` fallthrough (``NaN > 0`` and ``NaN < 0`` both false under ordinary IEEE-754 compare
semantics) is the *correct* target value, not just a plausible one -- confirmed to actually
compile and execute that way on real Ascend vector hardware by this round's M3 run (see
``round_1_result.md``).

**Integer dtypes -- a correctness requirement, not a performance one** (schedule doc §3): the
manifest's ``forall`` also admits 5 integer dtypes (int8/int16/int32/int64/uint8), served on
CUDA by ``SignIntFwdKernel(TorchFallbackKernel)``, which does not override ``Kernel.devices``
(stays ``frozenset({"cuda"})``) and is therefore ``unavailable()`` on an ``npu`` call. If
``AscendSignKernel.refusal()`` only accepted float dtypes, an ``npu`` + int call would find zero
eligible implementations and ``key_for`` would raise ``ValueError: no implementation serves this
call``. ``refusal()`` therefore accepts the full manifest domain (the 3 floats plus
``INT_DTYPES``, imported from ``tileops.kernels.elementwise.math_unary`` and not redefined
here), and ``forward()`` branches internally: floats run the TileLang program below; ints are
answered via ``torch.sign``, matching ``SignIntFwdKernel.handler`` bit-for-bit.

**Checked live, not assumed either way, a third distinct ACLNN coverage shape** (schedule doc
§3): a standalone pre-check script calling ``torch.sign`` directly on NPU-resident int8/int16/
int32/int64/uint8 tensors, cross-checked against a CPU ``torch.sign`` reference, found
``aclnnSign`` on this build natively implements only **int32/int64** among the 5 manifest int
dtypes --

```
torch.int8:  FAILED -- AclNN_Parameter_Error(EZ1001): self dtype DT_INT8 should be in
             [DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT64, DT_COMPLEX64,
              DT_COMPLEX128, DT_BOOL] or be BF16
torch.int16: FAILED -- same error, DT_INT16
torch.int32: OK, exact_match=True
torch.int64: OK, exact_match=True
torch.uint8: FAILED -- same error, DT_UINT8
```

-- a *third*, distinct shape from both already-documented siblings: wider than ``aclnnNeg``'s
gap (missing int16/uint8 only, int8 native) and present where ``aclnnAbs`` had none at all.
int8, int16, and uint8 all need the same shape of workaround ``AscendNegKernel`` already ships
for its own (smaller) gap: promote to int32, call ``torch.sign``, cast back down. This is
provably lossless here, more simply than Neg's case: ``torch.sign``'s output range is always
``{-1, 0, 1}`` regardless of input width, which trivially fits back into int8/int16/uint8 with
no truncation risk (uint8 in particular can never produce a ``-1`` in the first place, since an
unsigned source is never negative). Confirmed live under a second ``npu-guard`` reservation
(4096 random values per dtype spanning each dtype's full range, NPU
``torch.sign(x.to(int32)).to(dtype)`` vs. CPU ``torch.sign(x)``):

```
torch.int8:  promote-int32 path exact_match=True
torch.int16: promote-int32 path exact_match=True
torch.uint8: promote-int32 path exact_match=True
```

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
    # tuple SignIntFwdKernel.SUPPORTED_DTYPES already uses on the CUDA side (schedule doc §3).
    _FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    # The subset of INT_DTYPES whose width this NPU build's torch.sign (aclnnSign) actually
    # implements -- found by a live pre-check (module docstring), not assumed: aclnnSign's
    # error-reported support list only covers int32/int64 among this op's 5 int dtypes, a
    # third, distinct coverage shape from NegFwdOp's (int16/uint8 missing) and AbsFwdOp's
    # (no gap at all). int8/int16/uint8 need the int32 promote/truncate path in forward()
    # instead of a direct call.
    _NATIVE_NPU_SIGN_INT_DTYPES = (torch.int32, torch.int64)

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound path.

        Reuses ``relu.py``'s/``abs.py``'s round-2-calibrated helper verbatim (schedule doc
        §4): queries this SKU's real vector-core count (``get_num_vector_cores``, 72 on this
        SKU) and floors to the nearest power of two (64), falling back to 64 if the live query
        raises or returns a falsy value at build time. ``ReluFwdOp`` round 2's official
        (no-cache-flush) sweep already found 64/8192/2-stages the outright winner for this
        exact bandwidth-bound unary shape on this SKU, independently reconfirmed by
        ``BitwiseNotFwdOp`` and not beaten by ``NegFwdOp``'s/``AbsFwdOp``'s round-2 tile
        re-sweeps -- not re-swept here (schedule doc §4).

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
    # best-of-swept (ReluFwdOp round 2) and reconfirmed negative-sweep by NegFwdOp/AbsFwdOp
    # round 2 (schedule doc §4) -- not re-litigated here.
    _NUM_STAGES = 2

    # Tile for the persistent-grid path (schedule doc §4): round 1 ships tile=8192 uniformly
    # for all 3 float dtypes, the same value NegFwdOp/AbsFwdOp round 1 shipped and whose own
    # round-2 {4096, 2048} re-sweeps regressed every case on every dtype including float32.
    _TILE = 8192

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against (schedule
    # doc §1: "no decode case" in this op's own manifest workloads). This threshold matters for
    # test_sign_edge's n_total=4096 fixture, which is below tile=8192 (schedule doc §5).
    _SINGLE_BLOCK_THRESHOLD = 8192

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _sign_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = sign(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        A 3-way ``T.if_then_else`` comparison chain, matching ``SignFwdKernel.op_func``'s CUDA
        logical structure but **without** its fp32 promotion: a comparison against a
        dtype-native zero and a dtype-native ``{-1,0,1}`` select is exact at every finite
        fp16/bf16/fp32 value (schedule doc §2). ``T.cast(k, x_ub.dtype)`` for all three literals
        (not bare Python literals), mirroring ``relu.py``'s/``abs.py``'s ``T.cast(0,
        x_ub.dtype)`` so the comparison/select does not depend on literal-to-narrow-dtype
        promotion behavior.

        NaN falls through to ``zero``: ``NaN > 0`` and ``NaN < 0`` are both false under
        ordinary IEEE-754 compare semantics, matching ``torch.sign(nan) == 0`` (confirmed on
        CPU before assuming it held on Ascend vector hardware too -- schedule doc §2, module
        docstring).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): an undecorated helper's ``for i in
        T.Parallel(n):`` raises ``TypeError: 'ForFrame' object is not iterable`` under
        TileLang's eager builder (``dev_process.md`` §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero = T.cast(0, x_ub.dtype)
                one = T.cast(1, x_ub.dtype)
                neg_one = T.cast(-1, x_ub.dtype)
                y_ub[i] = T.if_then_else(
                    x_ub[i] > zero,
                    one,
                    T.if_then_else(x_ub[i] < zero, neg_one, zero),
                )

    @functools.lru_cache(maxsize=64)
    def _make_sign_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first path for small ``n_total``
        (``<= _SINGLE_BLOCK_THRESHOLD``). Required, not defensive, this round:
        ``test_sign_edge`` drives ``n_total=4096`` (float32), below ``tile=8192``
        (schedule doc §5).
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _sign_body(x_ub, y_ub, n_total, threads=min(2048, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_sign_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §4).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_sign_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_sign_generic_single_block`` instead. All 5 of this op's
        benchmarked cases, plus the correctness fixture's ``n_total=1_048_576``, hit this path.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_sign_persistent requires n_total ({n_total}) to divide "
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
                    _sign_body(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_sign_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide (schedule doc §5: "write it anyway for defensiveness/future shapes"). One
        block; a main loop over full tile-sized chunks, plus one smaller, statically-sized tail
        chunk. Not performance-tuned: no case or required test this round is expected to reach
        it (every shape this round exercises -- 1024 (int-only), 4096, 1_048_576, 2**24, 2**28
        -- either fits the single-block path, takes the int torch.sign branch, or divides
        ``num_blocks * tile`` exactly), mirroring ``relu.py``'s/``abs.py``'s
        ``_make_*_generic_single_block``.
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
                        _sign_body(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _sign_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_sign_kernel(n_total: int, dtype: str, tile: int):
        """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback."""
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            return _make_sign_single_block(n_total, dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_sign_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_sign_generic_single_block(n_total, dtype, tile)

    class AscendSignKernel(Kernel, UnaryElementwiseFwdInterface):
        """Sign on Ascend NPUs: ``y = sign(x)`` (-1, 0, or +1).

        Memory-bound (one load, a 2-compare/select chain, one store per element); exact at
        every finite value regardless of storage width, since the comparisons and the
        ``{-1,0,1}`` literals are all dtype-native (``docs/ascend950/dev/SignFwdOp/
        round_1_schedule.md`` §2). A fresh class per ``docs/ascend950/dev_process.md`` §1 -- it
        does not subclass the CUDA-locked ``UnaryKernel``/``FloatUnaryKernel`` family, which
        imports ``tilelang.language`` (the CUDA dialect facade).

        Accepts the op's *full* manifest domain -- the 3 benchmarked float dtypes plus the 5
        integer dtypes the manifest also admits -- not just the floats (schedule doc §3):
        ``forward()`` runs the TileLang program below for float input, and for integer input
        reproduces ``SignIntFwdKernel.handler``'s result bit-for-bit via ``torch.sign`` --
        direct for int32/int64 (this NPU build's ``aclnnSign`` implements only these two
        widths), via an int32 promote/sign/truncate round-trip for int8/int16/uint8
        (``aclnnSign`` does not implement those three directly on this build; see the module
        docstring) -- so an ``npu`` + int call still has an eligible implementation once this
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
            self._compiled_fn = (
                None
                if self._is_int
                else _make_sign_kernel(n_total, _dtype_str(dtype), _TILE)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                if input.dtype in _NATIVE_NPU_SIGN_INT_DTYPES:
                    # torch.sign directly -- bit-for-bit what SignIntFwdKernel.handler
                    # already computes for this dtype (schedule doc §3), and this NPU
                    # build's aclnnSign does implement this width (confirmed live, module
                    # docstring).
                    result = torch.sign(input)
                else:
                    # int8/int16/uint8: aclnnSign does not implement these widths on this
                    # NPU build (module docstring -- found by a live pre-check, not
                    # assumed). Promote to int32 (always wide enough, torch.sign's output
                    # range is always {-1, 0, 1} regardless of input width), sign there,
                    # and truncate back: bit-exact, since {-1, 0, 1} trivially fits back
                    # into int8/int16/uint8 with no truncation risk.
                    result = torch.sign(input.to(torch.int32)).to(input.dtype)
                # UnaryOp.forward() already made `input` contiguous before this call; the
                # extra .contiguous() is cheap defensive parity with
                # TorchFallbackKernel.forward()'s own explicit call, exercised directly by
                # test_compiled_non_contiguous_integer_fallback[SignFwdOp].
                return result.contiguous()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("SignFwdOp", "ascend", AscendSignKernel)

except ImportError:
    pass
