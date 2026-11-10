"""Ascend ReLU kernel: ``y = max(x, 0)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``ReluFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``ReluFwdKernel``
(``tileops.kernels.elementwise.activations.ReluFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``ReluFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Follows the tiling design in ``docs/ascend950/dev/ReluFwdOp/round_1_schedule.md``: this op is
memory-bound (one compare, one load, one store per element), so the only real knobs are
bulk-copy tile size, ``T.Pipelined`` stage count, and block/grid count, not the (trivial)
compute body.

Round 2 (``docs/ascend950/dev/ReluFwdOp/round_2_schedule.md`` / ``round_2_result.md``) queried
this SKU's real vector-core count (``tilelang.carver.arch.driver.get_num_vector_cores``) and
re-swept the persistent-grid path's parameters under the *official* (no-cache-flush) M4
measurement protocol -- which, surprisingly, **confirmed round 1's uncalibrated
``num_blocks=64``/``num_stages=2`` as already the best of everything tested**, not just an
unvalidated guess (see ``_calibrated_num_blocks``'s docstring for why an initial
cache-flushed scratch sweep pointed the wrong way). Round 2 separately replaced the
decode-size path's single-core ``T.Kernel(1)`` grid with a calibrated multi-block grid after
a cross-op finding (GeluFwdOp round 2) showed single-core decode kernels lose to per-core
launch-overhead serialization even when a single core's own data volume is trivial -- this
one *did* change behavior and fixed the decode case (see ``_DECODE_NUM_BLOCKS``).

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: ``docs/ascend950/dev_process.md`` §1 explicitly asks every op's worker to self-
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

        Round 1 copied ``num_blocks=64`` from ``example_vecadd.py``, uncalibrated against
        this SKU's real vector-core count, and missed its 80% latency target by 1.44x-1.59x
        on both prefill cases (``round_1_result.md``). Round 2
        (``docs/ascend950/dev/ReluFwdOp/round_2_result.md``) queried the real device via
        ``get_num_vector_cores`` -- ``vector_core_num=72`` on this SKU (device
        ``Ascend950DT_9592``), not a power of two, so ``_make_relu_persistent``'s exact
        ``n_total % (num_blocks * tile) == 0`` requirement (both prefill cases have
        ``n_total=2**23``) forces a power-of-two neighbor.

        **Important measurement-methodology lesson, not just a tuning result**: an initial
        sweep used ``tilelang.profiler.do_bench(backend="msprof_detail")``, which flushes the
        GM cache before every repeat call, and picked num_blocks=128 (oversubscribing the 72
        physical cores) plus num_stages=3 as fastest under that protocol. But the *official*
        M4 measurement (``tools.measure_baselines._time_reference``, used by
        ``measure_tileops_ascend.py`` and by every baseline number this op is judged against)
        does **not** flush cache between repeats -- every repeat reuses the same X/Y buffers,
        so later repeats can be served from this device's L2/system cache rather than HBM,
        which changes the bottleneck and flips the ranking. Re-measured with the official
        (no-flush) protocol, num_blocks=64/num_stages=2 (round 1's original, not 128/3) won
        outright: 11.46us vs 12.39us (128/2), 12.51us (64/3), 13.55us (128/3), 14.17us
        (256/2), 21.25us (32/2) -- i.e. *both* oversubscription (128, 256) and the extra
        pipeline stage (3) make it worse, not better, under the metric that actually
        determines the 80% target. This matches a cross-op finding (BitwiseNotFwdOp round 2):
        oversubscribing this SKU's real core count by a large margin measurably hurts a
        persistent-grid kernel. Round 2's net effect on this constant is therefore to
        *confirm* round 1's value via a live query and a corrected sweep, not to change it:
        floor (not ceiling) to the nearest power of two below the queried count, which
        recovers 64 for this SKU. Falls back to ``fallback`` if the live query raises or
        returns a falsy value at runtime, rather than crashing kernel build.

        Deliberately **not** evaluated at module import time: ``get_num_vector_cores`` queries
        the real device (``torch_npu.npu.get_device_properties``), so this is only called
        lazily, the first time a persistent-path kernel is actually built (always under an
        ``npu-guard`` reservation already, same as every other NPU touch in this file) --
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

    # Tile size and stage count for the memory-bound, large-n persistent-grid path (schedule
    # doc §3.1); num_blocks itself is resolved lazily by _calibrated_num_blocks() (see its
    # docstring for why).
    _TILE = 8192
    # Round-2 result doc: a corrected, official-protocol sweep (see _calibrated_num_blocks'
    # docstring) found num_stages=3 measurably *worse* than round 1's num_stages=2 (12.51us
    # vs 11.46us at num_blocks=64) -- kept at round 1's value, not changed.
    _NUM_STAGES = 2
    # Below this many elements, a persistent multi-block grid and T.Pipelined staging add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against
    # (schedule doc §3.2): run the whole call on a small multi-block grid instead (see
    # _DECODE_NUM_BLOCKS), one shot, no pipelining.
    _SINGLE_BLOCK_THRESHOLD = _TILE
    # Decode-size path (round-2 result doc): a cross-op finding from GeluFwdOp's round-2
    # Planner showed a decode-size kernel written as single-core T.Kernel(1) can lose purely
    # to serialized per-core launch/dispatch overhead, even though the per-core data volume
    # is trivial. Confirmed for ReLU by a direct sweep at this op's actual decode shape
    # (n_total=4096, bfloat16): num_blocks=1 (the old T.Kernel(1) path) measured 2.17us;
    # spreading across more blocks wins up to a point -- 8->1.37us, **16->1.28us (best,
    # just under the 1.2929us target)**, then degrades again as per-block launch overhead
    # outweighs further parallelism -- 32->1.35us, 64->1.61us, 128->2.83us, 256->5.37us.
    # This optimum (16) is well below this device's 72 real vector cores and is NOT derived
    # from get_num_vector_cores() like _NUM_BLOCKS above: at this data volume the governing
    # cost is fixed per-block overhead, not achievable bandwidth, so the right block count is
    # shape-specific and was found by direct measurement, not by matching the physical core
    # count. See round-2 result doc for the full table.
    _DECODE_NUM_BLOCKS = 16

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _relu_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = max(x_ub[:n], 0)``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``ReluFwdKernel.op_func`` (``kernels/elementwise/activations.py``) exactly:
        ``T.cast(0, x.dtype)`` on both branches, not a bare literal, so an fp16/bf16 tile does
        not depend on literal-to-narrow-dtype promotion behavior (schedule doc §3.1). No fp32
        promotion: ``max`` has no accumulation or overflow risk at any finite input.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): TileLang's eager builder rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into its trace-builder calls, so an undecorated
        helper's ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable`` -- it is calling the raw context-manager object instead of going through
        the builder's loop lowering.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                zero = T.cast(0, x_ub.dtype)
                y_ub[i] = T.if_then_else(x_ub[i] > zero, x_ub[i], zero)

    @functools.lru_cache(maxsize=64)
    def _make_relu_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first fallback for the decode-size
        path when ``n_total`` does not divide ``_DECODE_NUM_BLOCKS`` evenly (round 1's only
        decode-path kernel; round 2 found it ~1.7x slower than a multi-block grid at this op's
        actual decode shape -- see ``_make_relu_small_multi_block``, which is now the primary
        decode-size path).
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _relu_body(x_ub, y_ub, n_total, threads=min(128, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_relu_small_multi_block(n_total: int, dtype: str, num_blocks: int):
        """Multi-block, one-shot grid for the decode-size path (round-2 result doc).

        No ``T.Pipelined`` loop: each of ``num_blocks`` blocks does exactly one GM->UB copy,
        one ``_relu_body`` call, one UB->GM copy -- there is only one "iteration" per block by
        construction, so there is nothing to pipeline. Requires ``n_total`` to divide
        ``num_blocks`` exactly; the caller (``_make_relu_kernel``) only reaches this builder
        for a shape that does, and routes every other shape to ``_make_relu_single_block``
        instead.
        """
        assert n_total % num_blocks == 0, (
            f"_make_relu_small_multi_block requires n_total ({n_total}) to divide "
            f"num_blocks ({num_blocks}) exactly; the caller routes other shapes elsewhere"
        )
        chunk = n_total // num_blocks

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((chunk,), dtype)
                y_ub = T.alloc_shared((chunk,), dtype)
                begin = bx * chunk
                T.copy(X[begin : begin + chunk], x_ub)
                _relu_body(x_ub, y_ub, chunk, threads=min(2048, chunk))
                T.copy(y_ub, Y[begin : begin + chunk])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_relu_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3.1).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_relu_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_relu_generic_single_block`` instead (see that function's
        docstring for why -- schedule doc risk 1).
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_relu_persistent requires n_total ({n_total}) to divide "
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
                    _relu_body(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_relu_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide (schedule doc risk 1: "tail handling is unexercised, not proven unnecessary").

        One block; a main loop over full tile-sized chunks, plus one smaller, statically-sized
        tail chunk. ``n_total`` is already a Python int at kernel-build time (one compiled
        program per distinct ``(n_total, dtype)`` via ``entry_for``), so both the main-loop
        trip count and the tail chunk's width are ordinary Python ints baked into the generated
        program -- no dynamic-extent DMA is needed, which keeps this path's correctness
        independent of whatever dynamic-slice support this immature backend does or does not
        have. Not performance-tuned (single block, no cross-block parallelism): none of this
        op's three manifest cases reaches this function this round, so it is written but
        unexercised by M3/M4 -- see the round-1 result report.
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
                        _relu_body(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _relu_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_relu_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: small multi-block, single-block,
        persistent, or fallback."""
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            if n_total % _DECODE_NUM_BLOCKS == 0:
                return _make_relu_small_multi_block(n_total, dtype, _DECODE_NUM_BLOCKS)
            return _make_relu_single_block(n_total, dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * _TILE
        if n_total % block_span == 0:
            return _make_relu_persistent(n_total, dtype, num_blocks, _TILE, _NUM_STAGES)
        return _make_relu_generic_single_block(n_total, dtype, _TILE)

    class AscendReluKernel(Kernel, UnaryElementwiseFwdInterface):
        """ReLU on Ascend NPUs: ``y = max(x, 0)``.

        Memory-bound (negligible compute next to one load and one store per element); see
        ``docs/ascend950/dev/ReluFwdOp/round_1_schedule.md`` §3 for why the body stays in the
        storage dtype (no fp32 promotion) and for the tiling rationale. A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the
        CUDA dialect facade).
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._compiled_fn = _make_relu_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("ReluFwdOp", "ascend", AscendReluKernel)

except ImportError:
    pass
