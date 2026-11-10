"""Ascend NPU Rsqrt forward kernel: ``y = 1/sqrt(x)``, via the ``tilelang.ascend`` dialect.

Registers an additional implementation of ``RsqrtFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA ``RsqrtFwdKernel``
(``tileops.kernels.elementwise.math_unary.RsqrtFwdKernel``). No existing file is edited to do
this: ``tileops.backend.register_kernel_type`` joins this class to every ``RsqrtFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/RsqrtFwdOp/round_1_schedule.md`` (Planner, round 1) and
``docs/ascend950/dev/RsqrtFwdOp/round_2_schedule.md`` + ``round_2_result.md`` (round 2, the
per-dtype ``tile`` split below). This op is pure bandwidth-bound unary math (one load, one
``T.rsqrt`` call, one store per element) with no decode-sized manifest case -- all 5 benchmarked
cases are "16M"/"256M" prefill shapes. Round 1 reused the exact ``(num_blocks=64, tile=8192,
num_stages=2, threads=2048)`` config ``ReluFwdOp``, ``GeluFwdOp`` and ``BitwiseNotFwdOp``
independently converged on for the same cost shape on this SKU, for every dtype; it passed all
three 16M cases comfortably but missed both 256M cases (round_1_result.md §4). Round 2's
generated-source + ``msprof_detail`` pipe-occupancy diagnosis (round_2_result.md §2) found the
kernel DMA-bound (``aiv_mte2`` saturated well above ``aiv_vec`` at every measured scale) with no
VF-side conversion tax for either 16-bit dtype, and widened ``tile`` to 16384 for the two 2-byte
dtypes only (``_TILE_BY_DTYPE`` below) to amortize per-iteration DMA overhead; ``float32`` keeps
``tile=8192`` unchanged.

**The fp32 upcast before ``T.rsqrt`` is load-bearing, not a precision nicety**: reading
``IsSupportedSIMTUnaryMathTypeImpl`` in the Ascend codegen (``tilelang/src/ascend/codegen/
codegen_pto.cc``) shows ``T.SimtVF``'s math-intrinsic path only accepts f16/f32 -- bf16 is
rejected at that call site. The existing CUDA ``RsqrtFwdKernel.op_func`` already promotes
unconditionally (``T.rsqrt(T.cast(x, "float32"))``) for all three dtypes; mirroring that one
code path here (no dtype-conditional branching) sidesteps the bf16 restriction entirely, since
the value handed to ``T.rsqrt`` is always fp32 regardless of the UB buffer's storage dtype.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports -- same
deliberate omission ``relu.py``'s module docstring explains (several ops' workers write into
this same new directory concurrently; a shared aggregator file would be contended).
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
        """Persistent-grid block count for the pipelined, large-n path.

        Live-queries this SKU's real vector-core count (``get_num_vector_cores``, confirmed
        72 on ``Ascend950DT_9592`` by prior ops in this batch -- ``docs/ascend950/
        dev_process.md`` §7.5), floored to the nearest power of two (``floor_pow2(72) ==
        64``), rather than hardcoding the figure outright: cheap (~seconds), one
        ``npu-guard`` reservation, and a defensive check in case this session's
        hardware/environment differs. ``ReluFwdOp``/``GeluFwdOp``/``BitwiseNotFwdOp`` all
        independently landed on this same 64 after real on-device sweeps for the same
        one-load/one-store-per-element cost shape this op shares (round_1_schedule.md §4).

        Deliberately **not** evaluated at module import time -- merely importing this module
        must stay a no-NPU-touch no-op (``dev_process.md`` §4). ``lru_cache`` makes it a
        one-time query per process either way. Falls back to ``fallback`` if the live query
        raises or returns a falsy value at runtime, rather than crashing kernel build.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # Per-dtype-width tile size for the memory-bound, large-n persistent-grid path
    # (round_1_schedule.md §5.1, revised round_2_schedule.md §3.1). num_blocks itself is
    # resolved lazily by _calibrated_num_blocks() (see its docstring for why).
    #
    # Round 1 shipped one tile=8192 for every dtype and missed both 256M cases (round_1_
    # result.md §4); round 2's diagnostic pass (JITKernel.get_kernel_source() at 16M/256M
    # for all 3 dtypes, plus msprof_detail aiv_mte2/aiv_mte3/aiv_vec pipe occupancy at 16M/
    # 256M for bf16/fp16) found (round_2_result.md §2): (a) no half2/bfloat16x2<->float2
    # pack/unpack round trip in the generated per-thread body for either 16-bit dtype --
    # unlike TanhFwdOp, rsqrt's codegen lowers T.cast(..., "float32")/T.cast(..., dtype_str)
    # to a single scalar cast each ("(half)rsqrtf((float)x)"), structurally identical between
    # bf16 and fp16, so the "bf16<->fp32 is cheaper than fp16<->fp32" VF-tax hypothesis does
    # not apply here; (b) aiv_mte2 (the load/DMA pipe) is the dominant, most-saturated pipe
    # (87-93%) at every (dtype, scale) combination measured, well above aiv_vec (73-86%) --
    # i.e. this kernel is DMA-bound, not VF-compute-bound, at both scales. Doubling the
    # per-iteration transfer size for the two 2-byte dtypes only (float32 keeps tile=8192,
    # its own proven-safe configuration) is the DMA-side lever this evidence points at: it
    # halves the iteration count (and so the per-iteration DMA/sync overhead paid per byte)
    # without changing _rsqrt_body at all.
    #
    # UB footprint check (round_2_schedule.md §3.1, same arithmetic BitwiseNotFwdOp's round 1
    # used to find this hardware's 256 KiB UB overflow point): 2 buffers x 2 stages x tile x
    # dtype_bytes.
    #   float16/bfloat16: 2 x 2 x 16384 x 2 bytes = 131,072 B = 128 KiB (half of the 256 KiB
    #     failure point; same margin as the known-good int32/8192 config).
    #   float32 (unchanged): 2 x 2 x 8192 x 4 bytes = 131,072 B = 128 KiB (round 1's
    #     proven-safe footprint, untouched).
    # 32768-element tile for the 2-byte dtypes would hit exactly 256 KiB (BitwiseNotFwdOp's
    # confirmed overflow point) and is deliberately not used.
    _TILE_BY_DTYPE = {
        "float16": 16384,
        "bfloat16": 16384,
        "float32": 8192,
    }
    _NUM_STAGES = 2
    # T.SimtVF's hard 2048-thread cap (dev_process.md §7.5); tile exceeds it, so threads
    # stays independent of tile with T.Parallel spreading tile's iterations across lanes.
    _THREADS = 2048

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _rsqrt_body(x_ub, y_ub, n: int, dtype_str: str, threads: int) -> None:
        """``y_ub[:n] = rsqrt(x_ub[:n])``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``RsqrtFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        unconditional fp32 upcast before ``T.rsqrt``, cast back to the storage dtype at the
        boundary (code-style.md's promotion rule) -- this is load-bearing here, not just a
        precision nicety: ``T.SimtVF``'s math-intrinsic path rejects bf16 outright
        (round_1_schedule.md §3.2), and casting to fp32 first sidesteps that for all three
        dtypes via one code path, with no dtype-conditional branching.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not
        iterable`` (dev_process.md §7.5) -- it is calling the raw context-manager object
        instead of going through the builder's loop lowering.
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.rsqrt(T.cast(x_ub[i], "float32")), dtype_str)

    @functools.lru_cache(maxsize=64)
    def _make_rsqrt_single_block(n_total: int, dtype_str: str):
        """One block, one tile, no pipelining: correctness-only fallback for a small shape
        below ``_TILE_BY_DTYPE[dtype_str]`` (round_1_schedule.md §5.2).

        Serves exactly one correctness fixture this round --
        ``tests/ops/test_unary_math.py::test_rsqrt_edge`` (``n_total=4096``, fp32) -- which
        is not one of the 5 benchmarked manifest cases (no decode-labeled workload exists for
        this op) and carries no latency target. Modeled on ``relu.py``'s
        ``_make_relu_single_block`` / ``gelu.py``'s ``_build_single_shot``, not on the
        decode-multicore machinery those ops' round-2 *benchmarked* decode cases needed.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype_str)
                y_ub = T.alloc_shared((n_total,), dtype_str)
                T.copy(X, x_ub)
                _rsqrt_body(x_ub, y_ub, n_total, dtype_str, threads=min(_THREADS, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_rsqrt_persistent(n_total: int, dtype_str: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (round_1_schedule.md §5.1).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_rsqrt_kernel``) only reaches this builder for a shape that does. All 5
        benchmarked manifest cases and the 1,048,576-element bulk-correctness fixture are pure
        powers of two that divide the block span evenly for every dtype's ``_TILE_BY_DTYPE``
        entry (64*8192=2**19 for float32, 64*16384=2**20 for float16/bfloat16, round 2) -- no
        tail handling is implemented this round (none of this op's known test surface needs it).
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_rsqrt_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span
        threads = min(_THREADS, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _rsqrt_body(x_ub, y_ub, tile, dtype_str, threads)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_rsqrt_kernel(n_total: int, dtype_str: str):
        """Pick the tiling for ``n_total`` elements: persistent pipelined grid, or the
        small-shape single-block fallback.

        No decode case exists in this op's manifest (round_1_schedule.md §1/§7 point 4), so
        unlike ``ReluFwdOp``/``GeluFwdOp`` there is no multicore-spread path here -- only the
        one small non-benchmarked edge-case shape (4096 elements) needs the single-block
        fallback, and it carries no latency target to miss. ``tile`` is looked up per dtype
        (``_TILE_BY_DTYPE``, round_2_schedule.md §3.1) rather than a single shared constant.
        """
        num_blocks = _calibrated_num_blocks()
        tile = _TILE_BY_DTYPE[dtype_str]
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_rsqrt_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)
        if n_total <= tile:
            return _make_rsqrt_single_block(n_total, dtype_str)
        raise NotImplementedError(
            f"AscendRsqrtKernel: n_total={n_total} divides neither into "
            f"{num_blocks}x{tile} pipelined blocks nor fits the single-block fallback "
            f"(<= {tile} elements). round_1_schedule.md covers only the 5 benchmarked "
            "manifest cases plus the 1,048,576-element and 4096-element correctness "
            "fixtures; a general tail handler was deliberately not built this round."
        )

    class AscendRsqrtKernel(Kernel, UnaryElementwiseFwdInterface):
        """Rsqrt on Ascend NPUs: ``y = 1/sqrt(x)``.

        Always computes in fp32 regardless of storage dtype (``_rsqrt_body``), matching the
        in-tree CUDA ``RsqrtFwdKernel``'s own unconditional promotion -- a hard requirement
        on this backend, not just a precision nicety, since ``T.SimtVF``'s math-intrinsic
        path does not accept bf16 directly (see module docstring). A fresh class per
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
            self._compiled_fn = _make_rsqrt_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("RsqrtFwdOp", "ascend", AscendRsqrtKernel)

except ImportError:
    pass
