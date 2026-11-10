"""Ascend Tanh kernel: ``y = tanh(x)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``TanhFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``TanhFwdKernel``
(``tileops.kernels.elementwise.activations.TanhFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``TanhFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/TanhFwdOp/round_1_schedule.md``. This op is the best-precedented
one in the project so far -- ``T.tanh`` is already a confirmed-working Ascend intrinsic
(``GeluTanhAndMulFwdOp``, ``max_abs_err=0.0`` vs ``torch.tanh``), and the tiling here is a
verbatim, unmodified reuse of ``ReluFwdOp``'s already-calibrated persistent-grid constants
(``tile=8192``, ``num_blocks`` from the live-queried, floored-to-power-of-2 vector core count,
``num_stages=2``, ``threads=min(2048, tile)``): this op has the identical cost shape (1 load +
1 cheap transcendental + 1 store per element, memory-bound, 2 UB buffers). See
``_calibrated_num_blocks``'s docstring (copied from ``relu.py``) for why 64/2 is already the
validated operating point for this SKU and this op shape, not an unvalidated guess.

Unlike ``ReluFwdOp``/``SiluFwdOp``/``GeluFwdOp``, none of this op's 5 manifest cases is a
decode-sized shape -- all 5 are large, uniform, prefill-sized tensors that divide
``block_span = num_blocks * tile`` exactly (round_1_schedule.md §3.3), so the persistent
pipelined path is the only one actually graded this round. The small-shape paths below exist
only to keep ``tests/ops/test_activation.py``'s ``test_tanh_edge`` (``n_total=4096``) correct,
mirroring ``relu.py``'s own small-shape builders rather than inventing new ones.

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

        Verbatim reuse of ``relu.py``'s already-calibrated, already-hardware-validated
        helper (round_1_schedule.md §3.1): queries this SKU's real vector-core count
        (``get_num_vector_cores`` -- ``vector_core_num=72`` on ``Ascend950DT_9592``, not a
        power of two), then floors to the nearest power of two, landing on 64.

        ``ReluFwdOp`` round 2 already re-swept ``{32,64,128,256} x {num_stages 2,3}`` for
        this exact op shape (1 load + 1 cheap per-element op + 1 store, memory-bound) under
        the *official* (non-``msprof_detail``, no-cache-flush) ``measure_tileops_ascend.py``
        protocol and found 64/``num_stages=2`` win outright (11.46us vs 12.39us at 128/2,
        12.51us at 64/3, 13.55us at 128/3, 14.17us at 256/2, 21.25us at 32/2) -- both
        oversubscription and a third pipeline stage measurably hurt, not helped. This op's
        schedule (round_1_schedule.md §3.1) explicitly reuses that finding rather than
        re-sweeping it, since the cost shape is identical.

        Falls back to ``fallback`` if the live query raises or returns a falsy value at
        runtime, rather than crashing kernel build. Deliberately **not** evaluated at module
        import time -- ``get_num_vector_cores`` queries the real device, so this is only
        called lazily, the first time a persistent-path kernel is actually built (always
        under an ``npu-guard`` reservation already). ``lru_cache`` makes it a one-time query
        per process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Tile size and stage count for the memory-bound, large-n persistent-grid path
    # (schedule doc §3.1), reused unmodified from ReluFwdOp's already-calibrated values;
    # num_blocks itself is resolved lazily by _calibrated_num_blocks() (see its docstring).
    _TILE = 8192
    _NUM_STAGES = 2
    # Round 1 M4 iteration tried num_stages=3 (the Planner's one prescribed lever for a
    # miss, round_1_schedule.md §7) under the official measure_tileops_ascend.py protocol:
    # uniformly *worse* than 2 on all 5 cases (e.g. 16M-float32: 49.1us vs 45.6us; see
    # round_1_result.md §4) -- consistent with ReluFwdOp round 2's identical finding for
    # this same op shape. Kept at 2, not changed.
    # Below this many elements, a persistent multi-block grid and T.Pipelined staging add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against
    # (schedule doc §3.2): run the whole call on a small multi-block grid instead, one shot,
    # no pipelining. None of this op's 5 manifest cases reaches this threshold; it only
    # serves tests/ops/test_activation.py::test_tanh_edge (n_total=4096).
    _SINGLE_BLOCK_THRESHOLD = _TILE

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _tanh_body_native16(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = tanh(x_ub[:n])`` with no fp32 promotion -- ``float16``, UNUSED.

        **Not used by any builder below -- measured and rejected in round 2. Kept only as
        documented, reproducible negative evidence; do not wire this back in without new
        hardware evidence that contradicts round 2's measurement.**

        Round 2's compile-only investigation (``docs/ascend950/dev/TanhFwdOp/round_2_schedule.md``
        §1(a)) found that a direct, uncast ``T.tanh`` call on a ``float16`` buffer at this
        kernel's real tile geometry (``tile=8192``, ``threads=2048``, 4 elements/thread)
        compiles to 4 native ``htanh()`` calls operating in place on vectorized
        ``half2``-packed registers -- one 128-bit vectorized load, one 128-bit vectorized
        store, no ``half2<->float2`` unpack/repack and no local-array staging -- structurally
        leaner than ``_tanh_body_promote_f32``'s generated code for this dtype (confirmed by
        dumping ``JITKernel.get_kernel_source()`` for the real shipped kernel, not just the
        schedule's standalone probe -- round_2_result.md §2).

        **This structural prediction did not hold up on real hardware.** M4 (round_2_result.md
        §3) measured this macro, wired in for ``float16`` only, as *slower* than round 1's
        ``_tanh_body_promote_f32`` path at both manifest sizes, reproducibly across two
        independent ``npu-guard``-protected measurement passes and confirmed at the hardware
        AIV-cycle level (not just wall-clock): ~7.20-7.21M AIV cycles at 16M-float16 here vs.
        round 1's tightly-clustered ~6.96M (bfloat16/float32 cycle counts reproduced to within
        0.01% across every pass in both rounds, which is what makes this ~3.5% increase a real
        signal and not noise) -- and correspondingly ~8% slower in wall-clock ms at both 16M
        and 256M. The likely explanation: hardware ``htanh`` itself costs more cycles per call
        than the software ``tanhf`` path on this backend, enough to outweigh the saved
        conversion/staging instructions -- i.e. round 1's diagnosis (identical transcendental
        cost, only the surrounding casts differ) was an incomplete model; the transcendental
        call itself is not actually interchangeable in cost between the two dispatch paths.
        ``float16`` therefore stays on ``_tanh_body_promote_f32`` below, unchanged from round 1.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.tanh(x_ub[i])

    @T.macro
    def _tanh_body_promote_f32(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = tanh(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``TanhFwdKernel.op_func`` (``kernels/elementwise/activations.py``) exactly:
        promote to fp32 (code-style.md's overflow-prone-math rule -- though ``tanh`` is
        bounded in (-1, 1) for every finite input, so there is no overflow risk to design
        around here, unlike ``exp``/``sigmoid``), call ``T.tanh``, cast back to the storage
        dtype at the boundary. ``T.tanh`` is a confirmed-working Ascend intrinsic
        (``GeluTanhAndMulFwdOp``, ``max_abs_err=0.0`` vs ``torch.tanh``; round_1_schedule.md
        §2).

        Round 2 (``docs/ascend950/dev/TanhFwdOp/round_2_schedule.md``,
        ``round_2_result.md``) keeps this macro, unchanged from round 1, for **every** dtype:
        ``bfloat16`` because no safe native path compiles for it at all (see
        ``_tanh_body_native16``'s docstring), ``float32`` because it is a no-op here (compute
        dtype already equals storage dtype, so both ``T.Cast`` calls are optimized away
        entirely -- round_1_result.md §5), and ``float16`` because round 2's native-``htanh``
        alternative (``_tanh_body_native16``), despite compiling to structurally leaner code,
        measured *slower* on real hardware -- see that macro's docstring for the cycle-level
        evidence. Round 2 tried routing ``float16`` to the native macro and reverted it after
        M4 showed the regression.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): TileLang's eager builder rewrites only
        ``@T.prim_func``/``@T.macro`` bodies into its trace-builder calls, so an undecorated
        helper's ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable``.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.Cast(x_ub.dtype, T.tanh(T.Cast("float32", x_ub[i])))

    def _tanh_body_for_dtype(dtype: str):
        """Select the per-element compute macro for ``dtype``.

        A Python-level branch evaluated at kernel-build time (once per ``(n_total, dtype)``,
        memoized by each builder's own ``functools.lru_cache``), not a runtime branch inside
        the compiled kernel. Currently every dtype resolves to the same macro
        (``_tanh_body_promote_f32``) -- round 2 tried giving ``float16`` the native-``htanh``
        no-cast macro (``_tanh_body_native16``) instead, but reverted after M4 measured it as
        *slower* than the promote path on real hardware (see that macro's docstring and
        round_2_result.md §3). This function is kept (rather than calling
        ``_tanh_body_promote_f32`` directly from every builder) so a future round can
        reintroduce a per-dtype split here, in one place, if a real win is ever found.
        """
        del dtype  # every dtype currently uses the same macro; see docstring
        return _tanh_body_promote_f32

    @functools.lru_cache(maxsize=64)
    def _make_tanh_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first fallback for a small shape
        that does not divide ``_calibrated_num_blocks()`` evenly."""

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _tanh_body_for_dtype(dtype)(x_ub, y_ub, n_total, threads=min(128, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_tanh_small_multi_block(n_total: int, dtype: str, num_blocks: int):
        """Multi-block, one-shot grid for a small shape that divides ``num_blocks`` evenly
        (schedule doc §3.2): spreads the (correctness-only, non-latency-graded) edge case
        across the calibrated core count rather than collapsing to a single core, per the
        "decode/small shape shouldn't be single-core" lesson (dev_process.md §7.5) -- cheap
        to do right here even though this op's own manifest cases never reach this path.

        No ``T.Pipelined`` loop: each of ``num_blocks`` blocks does exactly one GM->UB copy,
        one dtype-selected compute-body call (``_tanh_body_for_dtype``), one UB->GM copy.
        Requires ``n_total`` to divide
        ``num_blocks`` exactly; the caller only reaches this builder for a shape that does.
        """
        assert n_total % num_blocks == 0, (
            f"_make_tanh_small_multi_block requires n_total ({n_total}) to divide "
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
                _tanh_body_for_dtype(dtype)(x_ub, y_ub, chunk, threads=min(2048, chunk))
                T.copy(y_ub, Y[begin : begin + chunk])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_tanh_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3.1); the
        primary path, serving all 5 manifest cases.

        UB budget: 2 buffers (x_ub, y_ub), not 3 -- ``2 x 2 x itemsize x 8192`` is 131,072
        bytes at fp32 and 65,536 bytes at fp16/bf16, both comfortably inside
        ``example_vecadd.py``'s own hardware-validated 3-buffer/fp32/196,608-byte footprint
        at the same tile/stage count, so no per-dtype tile shrinkage is needed (unlike
        ``SiluFwdOp``).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller only reaches
        this builder for a shape that does, and routes every other shape to
        ``_make_tanh_generic_single_block`` instead.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_tanh_persistent requires n_total ({n_total}) to divide "
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
                    _tanh_body_for_dtype(dtype)(x_ub, y_ub, tile, threads=min(2048, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_tanh_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide. One block; a main loop over full tile-sized chunks, plus one smaller,
        statically-sized tail chunk. Not performance-tuned: none of this op's 5 manifest
        cases reaches this function this round (all 5 divide ``block_span`` exactly), so it
        is written but unexercised by M3/M4 -- see the round-1 result report.
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
                        _tanh_body_for_dtype(dtype)(x_ub, y_ub, tile, threads=min(2048, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _tanh_body_for_dtype(dtype)(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_tanh_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements: small multi-block, single-block,
        persistent, or fallback."""
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            num_blocks = _calibrated_num_blocks()
            if n_total % num_blocks == 0:
                return _make_tanh_small_multi_block(n_total, dtype, num_blocks)
            return _make_tanh_single_block(n_total, dtype)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * _TILE
        if n_total % block_span == 0:
            return _make_tanh_persistent(n_total, dtype, num_blocks, _TILE, _NUM_STAGES)
        return _make_tanh_generic_single_block(n_total, dtype, _TILE)

    class AscendTanhKernel(Kernel, UnaryElementwiseFwdInterface):
        """Tanh on Ascend NPUs: ``y = tanh(x)``, promoted to fp32 then cast back.

        One load, one cheap transcendental call, one store per element; see
        ``docs/ascend950/dev/TanhFwdOp/round_1_schedule.md`` §3 for the tiling rationale (all
        tiling constants are unchanged since round 1). A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the
        CUDA dialect facade).

        Round 1 M4 found float16/bfloat16 were **not** memory-bound in practice despite moving
        half the bytes of float32: the generated per-thread body for a 16-bit storage dtype,
        promoted to fp32 for the ``tanh`` call and cast back (``_tanh_body_promote_f32``), pays
        for an extra ``half2<->float2``/``bfloat16x2<->float2`` pack/unpack round trip around
        the same ``tanhf`` call float32 makes with no conversion at all, and that cost
        dominates for an op this cheap -- float32 passed its 80% target (barely), both
        float16/bfloat16 prefill cases missed it by ~2.05x-3.04x (round_1_result.md §5).

        Round 2 (``docs/ascend950/dev/TanhFwdOp/round_2_schedule.md``) investigated an
        asymmetric fix based on compile-only probes: give ``float16`` a direct, uncast
        ``T.tanh`` call, which dispatches to a native ``htanh`` on vectorized ``half2``
        registers with no pack/unpack (``_tanh_body_native16``), while leaving ``bfloat16`` on
        the promote path (no safe native or intermediate-cast alternative exists for it --
        round_2_schedule.md §1(b)/(c)). **The float16 half of this did not pan out**: wired in
        and measured on real hardware (round_2_result.md §3), the structurally leaner native
        path was *slower* than ``_tanh_body_promote_f32``, reproducibly, at both AIV-cycle and
        wall-clock granularity -- i.e. the hardware ``htanh`` instruction itself appears to cost
        more cycles per call than the software ``tanhf`` path, enough to outweigh the removed
        conversion overhead. Round 2 reverted ``float16`` to ``_tanh_body_promote_f32``, so
        every dtype now runs the identical, round-1 compute body again; see
        ``_tanh_body_native16``'s docstring for the rejected alternative and its evidence.
        Both float16 and bfloat16 are expected to keep missing the 80% target with no further
        kernel-level lever identified for either.
        """

        devices = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype):
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._compiled_fn = _make_tanh_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("TanhFwdOp", "ascend", AscendTanhKernel)

except ImportError:
    pass
