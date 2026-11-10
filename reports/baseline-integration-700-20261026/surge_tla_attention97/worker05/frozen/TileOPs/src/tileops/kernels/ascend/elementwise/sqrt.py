"""Ascend sqrt kernel: ``y = sqrt(x)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``SqrtFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``SqrtFwdKernel``
(``tileops.kernels.elementwise.math_unary.SqrtFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``SqrtFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/SqrtFwdOp/round_1_schedule.md`` (Planner, round 1). ``sqrt`` is a
single-intrinsic, one-load/one-compute/one-store unary op -- computationally closest to
``ReluFwdOp``/``BitwiseNotFwdOp`` in this project's precedent set, not to the heavier
``GeluFwdOp``/``SiluFwdOp``. The manifest's two workloads (16M = 4096x4096, 256M =
16384x16384 elements) are both large, uniform, prefill-sized tensors -- no decode-shaped
(``[1, N]``) case exists for this op, so (per the schedule doc §3) this file deliberately does
**not** implement the small-n_total core-spreading machinery ``relu.py``/``gelu.py``/
``silu.py`` need for their own decode cases.

Tiling (schedule doc §3) directly reuses ``relu.py``'s round-1/round-2-confirmed-winning
persistent-grid configuration: ``num_blocks`` live-queried via
``tilelang.carver.arch.driver.get_num_vector_cores`` (expected 72 on this SKU) and floored to
the nearest power of two (64, since none of this op's shapes -- all pure powers of two -- are
evenly divisible by 72 = 2^3 x 3^2), ``num_stages=2``, ``threads=min(2048, tile)=2048``.
``tile=8192`` for bfloat16/float32; float16 uses the schedule doc §3 fallback lever
(``tile=16384``) after M4 found the uniform-8192 config missed its 256M case -- see
``_WIDE_TILE_DTYPES`` below and round_1_result.md for the measurement history.

Math body mirrors ``SqrtFwdKernel.op_func`` (``tileops/kernels/elementwise/math_unary.py``):
compute in fp32 regardless of storage dtype, cast back at the boundary
(``code-style.md``'s promotion rule, and the hard compile-time constraint that ``T.SimtVF``'s
math intrinsics only support f16/f32 at the codegen level, not bf16 --
``dev_process.md`` §7.5, ``IsSupportedSIMTUnaryMathTypeImpl``). This also makes the
negative-domain edge case (``test_sqrt_edge`` feeds ``-1.0``) come out as NaN, matching
``torch.sqrt``: confirmed on real hardware by a standalone single-block confirmatory run
before this file was written (see round_1_result.md) -- the underlying ``sqrtf``/``hsqrt``/
``sqrt`` dispatch is a standard libm-style IEEE-754 sqrt with no special-casing needed.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports (otherwise
required by ``TileOPs/.claude/rules/code-style.md`` for every ``src/tileops/kernels/*``
subpackage): ``docs/ascend950/dev_process.md`` §1 explicitly asks every op's worker to self-
register from its own file and *not* build a shared aggregator file under ``kernels/ascend/``,
since several ops' workers write into this same new directory concurrently and a shared
``__init__.py`` would be a file they all contend for. ``kernels/ascend`` and
``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420) namespace packages.
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
    from tileops.kernels.kernel_base import Entry, Kernel

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound path.

        Every shape this op is measured or tested at (the 2^20-element correctness fixture,
        the 2^24/2^28-element benchmark cases) is a pure power of two, while this SKU's real
        vector-core count (``get_num_vector_cores``, expected 72 = 2^3 x 3^2) cannot divide any
        of them evenly -- the same constraint ``BitwiseNotFwdOp``/``ReluFwdOp`` independently
        hit. Floors (not ceils) to the nearest power of two, which both of those ops'
        official-protocol sweeps found beats the next power of two up (64 beats 128) for this
        family of cheap, bandwidth-bound unary kernels. Falls back to ``fallback`` if the live
        query raises or returns a falsy value at runtime, rather than crashing kernel build.

        Deliberately **not** evaluated at module import time: ``get_num_vector_cores`` queries
        the real device, so this is only called lazily, the first time a persistent-path kernel
        is actually built (always under an ``npu-guard`` reservation already) -- merely
        importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
        ``lru_cache`` makes it a one-time query per process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Tile size and stage count for the memory-bound, large-n persistent-grid path (schedule
    # doc §3): num_blocks itself is resolved lazily by _calibrated_num_blocks() (see its
    # docstring for why). Directly reuses relu.py's round-1/round-2-confirmed values: round 2's
    # corrected, official-protocol (no-cache-flush) sweep found num_stages=3 measurably worse
    # than num_stages=2 for this same family of cheap bandwidth-bound unary ops, so this file
    # does not default to 3 just because a heavier op (silu.py) uses it.
    _TILE = 8192
    _NUM_STAGES = 2
    # round_1_result.md: the primary uniform-tile config (8192 for every dtype) passed all
    # three 16M cases and 256M-bfloat16, but missed 256M-float16 (measured 0.431082ms vs a
    # 0.408941ms target, ratio 0.759 -- the official measure_tileops_ascend.py protocol, not
    # msprof_detail). The schedule doc §3 fallback lever (wider tile for fp16/bf16, mirroring
    # silu.py's _tile_elems_for_dtype) was first tried for *both* fp16 and bf16 together: that
    # closed the 256M-float16 gap (0.431082ms -> 0.330501ms) but regressed 16M-bfloat16 from a
    # comfortable pass (0.016099ms, ratio 0.848) to a miss (0.018485ms, ratio 0.739) -- bf16 was
    # never the dtype that needed widening (its 256M case already passed at 8192, ratio 0.834).
    # Scoped the lever to float16 only, leaving bfloat16 at the narrow tile that already passed
    # both of its cases; see round_1_result.md for the full before/after table. fp32 stays at
    # 8192 throughout (its only case passed comfortably from the start). Kept num_stages=2
    # rather than silu.py's 3: this op's body is far cheaper than silu's, and relu.py's
    # official-protocol sweep already found 3 stages worse for this same cheap-unary-op family.
    _WIDE_TILE = 16384
    _WIDE_TILE_DTYPES = frozenset({"float16"})

    def _tile_for_dtype(dtype_str: str) -> int:
        return _WIDE_TILE if dtype_str in _WIDE_TILE_DTYPES else _TILE

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against: run the
    # whole call on a single block instead, one shot, no pipelining. No decode-shaped manifest
    # case exists for this op (schedule doc §3), so this threshold only needs to cover the
    # correctness/edge fixtures' small n_total values (4096, 1_048_576), not a tuned decode path.
    # Deliberately keyed off the narrower (fp32) tile, not the per-dtype one, so routing to the
    # single-block path does not itself depend on dtype.
    _SINGLE_BLOCK_THRESHOLD = _TILE

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _sqrt_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = sqrt(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``SqrtFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        compute in fp32 regardless of storage dtype, cast back to ``x_ub.dtype`` at the
        boundary. This promotion is not just a precision nicety here -- it is a hard compile-
        time requirement for bf16 inputs, since ``T.SimtVF``'s math intrinsics only recognize
        f16/f32 element types at the codegen level (``dev_process.md`` §7.5). It also makes the
        negative-domain edge case come out as NaN (confirmed on real hardware against
        ``torch.sqrt`` before this kernel was written -- see round_1_result.md), with no
        special-casing needed: the underlying ``sqrtf``/``hsqrt``/``sqrt`` dispatch is a
        standard libm-style IEEE-754 sqrt.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): TileLang's eager builder rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into its trace-builder calls, so an undecorated
        helper's ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable``.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.sqrt(T.cast(x_ub[i], "float32")), x_ub.dtype)

    @functools.lru_cache(maxsize=64)
    def _make_sqrt_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first path for small ``n_total``
        (this op's edge-case fixture, ``n_total=4096``) that does not warrant a persistent
        multi-block grid.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _sqrt_body(x_ub, y_ub, n_total, threads=min(2048, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_sqrt_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_sqrt_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_sqrt_generic_single_block`` instead.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_sqrt_persistent requires n_total ({n_total}) to divide "
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
                    _sqrt_body(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_sqrt_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide. One block; a main loop over full tile-sized chunks, plus one smaller,
        statically-sized tail chunk. Not performance-tuned: none of this op's manifest cases or
        test fixtures reach this function this round (every one of 4096, 1_048_576, 2^24, 2^28
        either fits the single-block path or divides ``num_blocks * tile`` exactly) -- written
        for ``UnaryElementwiseFwdInterface``'s general contract (an arbitrary shape, not just
        the benchmarked ones), mirroring ``relu.py``'s ``_make_relu_generic_single_block``.
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
                        _sqrt_body(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _sqrt_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_sqrt_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback."""
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            return _make_sqrt_single_block(n_total, dtype)
        tile = _tile_for_dtype(dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_sqrt_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_sqrt_generic_single_block(n_total, dtype, tile)

    class AscendSqrtKernel(Kernel, UnaryElementwiseFwdInterface):
        """Sqrt on Ascend NPUs: ``y = sqrt(x)``.

        Computes in fp32 regardless of storage dtype and casts back at the boundary
        (``docs/ascend950/dev/SqrtFwdOp/round_1_schedule.md`` §2); ``torch.sqrt(-1.0) == nan``
        is reproduced exactly, with no special-casing, because the underlying intrinsic is a
        standard IEEE-754 sqrt. A fresh class per ``docs/ascend950/dev_process.md`` §1 -- it
        does not subclass the CUDA-locked ``UnaryKernel``/``FloatUnaryKernel`` family, which
        imports ``tilelang.language`` (the CUDA dialect facade).
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._compiled_fn = _make_sqrt_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("SqrtFwdOp", "ascend", AscendSqrtKernel)

except ImportError:
    pass
