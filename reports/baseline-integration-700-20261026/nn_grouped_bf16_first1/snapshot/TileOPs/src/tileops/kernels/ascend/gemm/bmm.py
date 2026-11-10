"""Ascend batched GEMM kernel: ``d[i] = a[i] @ b[i]`` for every batch item, via the
``tilelang.ascend`` dialect.

Registers an additional implementation of ``BmmFwdOp``'s ``BmmFwdInterface`` under the
``"ascend"`` dispatch key, beside the in-tree CUDA ``BmmKernel``/``BmmPersistentKernel``
(``tileops.kernels.gemm``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``BmmFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

This is the project's first Cube/matmul op (batch 1 was all Vector/AIV elementwise), so
it does not reuse any batch-1 convention beyond the file layout and the self-registration
mechanics. Follows ``docs/ascend950/dev/BmmFwdOp/round_1_schedule.md`` (tiling design,
persistent-scheduler choice, the B-operand layout question below).

**The one real open technical risk (schedule §3), resolved**: ``b`` arrives as
``[B, K, N]``, N-innermost (``torch.bmm``'s native layout), but ``T.gemm(...,
transpose_B=True)`` -- the only ``T.gemm`` calling convention this backend has any
working precedent for (schedule §3.2: zero precedent anywhere in this codebase for
``transpose_B=False``) -- needs its B operand shaped ``[N, K]`` at L0. The schedule's
primary plan (do the transpose at the GM->L1 copy, which ``copy_op.py``'s own docstring
documents as supported: "transpose (bool): Ascend GM->L1 only") was verified correct in a
standalone smoke test before this file was written (6/6 cases passed, both dtypes,
non-128-aligned shapes included -- see round_1_result.md) and is what ``_make_bmm_kernel``
below does: ``l1b`` is allocated directly as ``(BLOCK_N, BLOCK_K)`` and loaded with
``T.copy(..., transpose=True)`` straight from the ``[K, N]``-shaped GM region; the
L1->L0 copy for B is then a plain, non-transposing copy. This avoids the
docstring-contradicting L1->L0 transpose path that ``examples/ascend/deepgemm``'s (
currently non-importable in this venv) ``make_l1_to_l0_copy`` uses instead.

**The scheduler risk (schedule §2/§8 risk 2)**: ``T.AscendBatchedTileScheduler`` had no
confirmed on-device run anywhere in this workspace before this op (its only usage
precedent, deepgemm's ``kernels/mnk/bf16_gemm.py``, cannot currently be imported/run --
missing Cython extension, see the schedule). A standalone smoke test (plain ``T.Kernel``,
not ``T.MixedKernel``; ``T.Pipelined`` for the K loop, not ``T.Task()``/AutoSchedule --
lower risk, per schedule §4.2) confirmed the scheduler's stateful walk
(``init``/``valid``/``tile``/``batch``/``next_block``) works correctly under this
simpler harness before being adopted here, so round 1 uses it directly rather than
falling back to a hand-written swizzle loop -- every one of this op's 16 manifest cases
has M and N exactly divisible by ``BLOCK_M``/``BLOCK_N`` (128), so the scheduler's tail
handling and GROUP_M swizzle are exercised only in the trivial (single-tile-per-row)
regime, not stress-tested here.

Round 1 used ``BLOCK_M = BLOCK_N = BLOCK_K = MAD_M = MAD_N = MAD_K = 128`` (a single
``T.gemm`` call per K-step, no K sub-split) and measured 12/16 M4 passes (corrected
formula, see ``docs/ascend950/dev/BmmFwdOp/round_1_result.md``); the 4 misses were
exactly its 4 cases with the most K-reduction work per output tile (K=2048 x2, K=4096,
and ``moe-prefill``'s K=2048 combined with the largest tile count of any case).

**Round 2 tried decoupling ``BLOCK_K`` from ``MAD_K`` (schedule
``docs/ascend950/dev/BmmFwdOp/round_2_schedule.md`` §2) and reverted it** after the
official ``measure_tileops_ascend.py`` protocol showed it to be a net regression, not an
improvement -- see ``docs/ascend950/dev/BmmFwdOp/round_2_result.md`` for the full 16-case
table. In short: widening ``BLOCK_K`` to 256 (``MAD_K`` fixed at 128, two MAD substeps
per L1 reload via a plain unrolled Python loop) dropped the M4 pass count from 12/16 to
10/16 -- two previously-passing cases (``square-1k``, both dtypes) regressed into misses,
while **none** of the original 4 misses closed (``square-2k`` got slightly worse;
``square-4k``/``moe-prefill`` improved only 1-3%, nowhere near the 6-22% needed). A
follow-up probe of the schedule's ``moe-prefill``-specific ``BLOCK_K=512`` stretch was
worse still (0.520704 ms vs 256's 0.480438 ms vs round 1's own 0.496573 ms) --
latency rose *monotonically* with ``BLOCK_K`` on this kernel, the opposite of the
schedule's hypothesis. Suspected mechanism (not confirmed via ``msprof_detail`` pipe
occupancy, which round 2's time budget did not reach): the inner MAD-substep loop is a
plain Python ``for`` (unrolled at trace time, no ``T.Pipelined``/``T.serial`` wrapping),
so sibling substeps never overlap their L1->L0 copy against the previous substep's
``T.gemm`` -- widening ``BLOCK_K`` trades away ``T.Pipelined``'s cross-K-block
GM->L1/compute overlap (fewer, larger outer iterations) for a longer, fully-serial
per-block substep chain that nothing hides. This file ships round 1's kernel unchanged
(``BLOCK_M = BLOCK_N = BLOCK_K = MAD_M = MAD_N = MAD_K = 128``, no K sub-split); the
``BLOCK_K``/``MAD_K`` decoupling is recorded as a tried-and-rejected lever, not a
TODO, for any future round 3 on this op.

**A genuine backend-immaturity finding survives the revert, recorded here for other ops'
benefit**: round 2's (reverted) inner-substep code computed its per-substep
``clear_accum`` flag with ``if k_mad_sub == 0: is_first = (k_block_idx == 0)\\nelse:
is_first = False`` inside the ``@T.prim_func`` body, and this silently produced 97%+
wrong output elements (max abs err ~285 against ``torch.bmm``) *even with only one
substep*, where the ``else`` branch is never reached at trace time. Replacing it with
the equivalent ternary **expression** ``(k_block_idx == 0) if k_mad_sub == 0 else
False`` was bit-exact. Isolated via standalone repro scripts (not kept in-tree,
documented verbatim in ``round_2_result.md``) to ``T.prim_func``'s tracer treating a
Python ``if``/``else`` *statement* specially -- apparently regardless of whether its
condition is a compile-time Python constant -- in a way that corrupts the enclosing
scope's L0 buffer-version/double-buffering analysis. Any future kernel on this backend
that branches on a trace-time-constant condition inside a ``T.Pipelined``-pipelined
region should prefer a ternary expression over an ``if`` statement until this is
understood/fixed upstream.

Epilogue: no Vector engine step (schedule §5 -- this op's compute is 100% Cube). The L0C
fp32 accumulator is copied straight to the ``[B, M, N]`` GM output tile, which performs
the fp32->fp16/bf16 narrowing cast as part of the copy itself (``example_gemm.py``'s
non-``MIXED`` epilogue pattern).

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a
CUDA-only process) is a no-op: the Ascend-specific imports, the kernel class, and the
registration call are wrapped in one ``try/except ImportError`` block, per
``docs/ascend950/dev_process.md`` §1.

No subpackage ``__init__.py`` with ``__all__`` re-exports: ``docs/ascend950/dev_process.md``
§1 explicitly asks each op's worker to self-register from its own file, not build a shared
aggregator several concurrent workers would contend for. ``kernels/ascend`` and
``kernels/ascend/gemm`` are left as implicit (PEP 420) namespace packages, matching
``kernels/ascend/elementwise`` from batch 1.
"""

import functools
from typing import Optional

import torch

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_cube_cores

    import tileops.backend
    from tileops.kernels.gemm import BmmCall, BmmFwdInterface
    from tileops.kernels.kernel_base import Entry, Kernel

    # Single MAD tile per block on every axis (schedule §4.2, simplified -- see module
    # docstring for round 2's BLOCK_K=256/MAD_K=128 split, tried and reverted as a net
    # regression under the official measurement protocol). A value every kernel build
    # call must agree on (the scheduler, the L1/L0 allocations and the K loop all read
    # it), so it is a module constant, not a per-call local.
    _BLOCK_M = 128
    _BLOCK_N = 128
    _BLOCK_K = 128
    # Double-buffer both L1 and L0 stages for the K loop's T.Pipelined overlap.
    _NUM_STAGES = 2

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_cube_cores(fallback: int = 36) -> int:
        """This SKU's real Cube core count, queried live (not hardcoded).

        ``docs/ascend950/dev_process.md`` §7.5: the confirmed value for
        ``Ascend950DT_9592`` is 36 (``tilelang.carver.arch.driver.get_num_cube_cores``,
        matching ``torch_npu.npu.get_device_properties(0).cube_core_num``), queried live by
        this op's Planner in the same session. Queried again here (not hardcoded) so a
        kernel built on a different SKU still gets its own real core count; falls back to
        the confirmed value for this box if the live query raises or returns a falsy value,
        rather than crashing kernel build.

        Deliberately **not** evaluated at module import time: merely importing this module
        must stay a no-NPU-touch no-op (``dev_process.md`` §4). Only called lazily, the
        first time a kernel is actually built -- always under an ``npu-guard`` reservation
        already, same as every other NPU touch in this project. ``lru_cache`` makes it a
        one-time query per process either way.
        """
        try:
            queried = get_num_cube_cores(device_id=0)
        except Exception:
            queried = None
        return int(queried) if queried else fallback

    @functools.lru_cache(maxsize=64)
    def _make_bmm_kernel(batch: int, m: int, n: int, k: int, dtype: str):
        """Build (and cache) the persistent batched-GEMM kernel for one ``(batch, m, n, k,
        dtype)`` combination.

        Grid: ``min(real_cube_core_count, total_output_tiles)`` persistent cores, each
        walking ``T.AscendBatchedTileScheduler``'s global ``(batch, m_tile, n_tile)`` grid
        (schedule §4.2). ``BLOCK_M``/``BLOCK_N`` both 128; every one of this op's 16
        manifest cases has M and N exactly divisible by 128 (round-1 scope -- a shape that
        is not would still be handled correctly via the scheduler's own ``actual_m``/
        ``actual_n`` tail clamp, just untested by this op's cases).
        """
        block_m, block_n, block_k = _BLOCK_M, _BLOCK_N, _BLOCK_K
        m_tiles = -(-m // block_m)
        n_tiles = -(-n // block_n)
        num_k_blocks = -(-k // block_k)
        total_tiles = batch * m_tiles * n_tiles
        num_blocks = min(_calibrated_num_cube_cores(), total_tiles)

        @T.prim_func
        def main(
            A: T.Tensor((batch, m, k), dtype),
            B: T.Tensor((batch, k, n), dtype),
            D: T.Tensor((batch, m, n), dtype),
        ):
            with T.Kernel(num_blocks) as bx:
                sched = T.AscendBatchedTileScheduler(
                    block_m=block_m,
                    block_n=block_n,
                    num_cores=num_blocks,
                    shape_m=m,
                    shape_n=n,
                    num_groups=batch,
                )
                a_l1 = T.alloc_l1((block_m, block_k), dtype)
                # Allocated directly as [N, K] (the transposed shape): the GM->L1 copy
                # below transposes on the way in, so no further transpose is needed before
                # T.gemm(..., transpose_B=True) (module docstring; schedule §3.3 primary
                # plan).
                b_l1 = T.alloc_l1((block_n, block_k), dtype)
                a_l0 = T.alloc_l0a((block_m, block_k), dtype)
                b_l0 = T.alloc_l0b((block_n, block_k), dtype)
                res = T.alloc_l0c((block_m, block_n), "float32")
                T.annotate_buffer_versions(
                    {a_l1: _NUM_STAGES, b_l1: _NUM_STAGES, a_l0: _NUM_STAGES, b_l0: _NUM_STAGES}
                )
                sched.init(bx)
                while sched.valid():
                    batch_idx = sched.batch()
                    m_idx, n_idx, actual_m, actual_n = sched.tile()
                    for k_block_idx in T.Pipelined(num_k_blocks, num_stages=_NUM_STAGES):
                        k_idx = k_block_idx * block_k
                        actual_k = T.min(k - k_idx, block_k)
                        T.copy(
                            A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + actual_k],
                            a_l1[0:actual_m, 0:actual_k],
                        )
                        # GM [K, N] slice -> L1 [N, K]: the one transpose this kernel does,
                        # at the documented-supported GM->L1 stage.
                        T.copy(
                            B[batch_idx, k_idx : k_idx + actual_k, n_idx : n_idx + actual_n],
                            b_l1[0:actual_n, 0:actual_k],
                            transpose=True,
                        )
                        T.copy(a_l1[0:actual_m, 0:actual_k], a_l0[0:actual_m, 0:actual_k])
                        # Plain copy: b_l1 is already [N, K], b_l0 wants [N, K].
                        T.copy(b_l1[0:actual_n, 0:actual_k], b_l0[0:actual_n, 0:actual_k])
                        T.gemm(
                            a_l0[0:actual_m, 0:actual_k],
                            b_l0[0:actual_n, 0:actual_k],
                            res[0:actual_m, 0:actual_n],
                            transpose_B=True,
                            clear_accum=(k_block_idx == 0),
                        )
                    # No Vector epilogue (schedule §5): the fp32->dtype narrowing cast
                    # happens as part of this one L0C->GM copy.
                    T.copy(
                        res[0:actual_m, 0:actual_n],
                        D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                    )
                    sched.next_block()

        return tilelang.compile(main, target="ascend", out_idx=-1)

    # Aligned L1-direct candidate: the same persistent scheduler and outer K pipeline
    # as ``_make_bmm_kernel``, but the MAD tile is handed L1 operands directly, so the
    # backend's ``tl.ascend_gemm_l1`` template owns the L0A/L0B sub-K pipeline
    # (``tilelang/src/tl_templates/ascend/gemm.h``: two internal L0 slots, M_MTE1
    # event ids 0/1, L1->L0 copies alternating with MAD).
    #
    # This is deliberately a *separate* cached builder, not a mutation of
    # ``_BLOCK_*``/``_NUM_STAGES``: the untouched L0 body above remains the route for
    # every M/N/K not exactly divisible by 128. Round 3's real smoke failure of this
    # mechanism on (4, 200, 300, 128, float16) (max abs err 84.3125) is why eligibility
    # requires M%128 == N%128 == K%128 == 0; a passing aligned tile is not a tail proof.
    #
    # L1 A+B footprint = (block_m + block_n) * block_k * itemsize * num_stages, below
    # 384 KiB for the finite grid used this round (K tile 128/256, 1/2/3 stages).
    # L0A/L0B (32 KiB per internal slot) and L0C (64 KiB fp32 128x128) are allocated
    # inside the lowering/header, not here; the grid is the requested Cube count capped
    # by the real output tiles, never the 64-Vector-core default.
    _L1_BLOCK_K = (128, 256)
    _L1_NUM_STAGES = (1, 2, 3)
    _MAX_CUBE_BLOCKS = 36

    @functools.lru_cache(maxsize=64)
    def _make_bmm_aligned_l1_kernel(
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: str,
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        num_blocks: int,
    ):
        """Build (and cache) one aligned L1-direct GEMM candidate.

        Eligibility is deliberately narrower than the public op: M/N/K must be exact
        multiples of the fixed 128x128 tile and K must divide the requested K tile, so
        every T.Pipelined iteration hands the static-region GEMM a complete,
        initialized tile. Anything else must keep using ``_make_bmm_kernel``.
        """
        if block_m not in (128, 256) or block_n not in (128, 256):
            raise ValueError(
                f"block_m/block_n must be 128 or 256, got {block_m}/{block_n}")
        if m % block_m or n % block_n or k % _BLOCK_K:
            raise ValueError(
                "aligned L1 builder requires M, N and K divisible by "
                f"{_BLOCK_K}, got M={m}, N={n}, K={k}"
            )
        if block_k not in _L1_BLOCK_K:
            raise ValueError(f"block_k must be one of {_L1_BLOCK_K}, got {block_k}")
        if k % block_k:
            raise ValueError(f"block_k={block_k} must divide K={k}")
        if num_stages not in _L1_NUM_STAGES:
            raise ValueError(f"num_stages must be one of {_L1_NUM_STAGES}, got {num_stages}")
        if not 1 <= num_blocks <= _MAX_CUBE_BLOCKS:
            raise ValueError(
                f"num_blocks must be in [1, {_MAX_CUBE_BLOCKS}], got {num_blocks}"
            )

        m_tiles = m // block_m
        n_tiles = n // block_n
        num_k_blocks = k // block_k
        total_tiles = batch * m_tiles * n_tiles
        grid = min(num_blocks, total_tiles)
        if grid < 1:
            raise ValueError("aligned L1 builder has an empty output tile grid")

        @T.prim_func
        def main(
            A: T.Tensor((batch, m, k), dtype),
            B: T.Tensor((batch, k, n), dtype),
            D: T.Tensor((batch, m, n), dtype),
        ):
            with T.Kernel(grid) as bx:
                sched = T.AscendBatchedTileScheduler(
                    block_m=block_m,
                    block_n=block_n,
                    num_cores=grid,
                    shape_m=m,
                    shape_n=n,
                    num_groups=batch,
                )
                a_l1 = T.alloc_l1((block_m, block_k), dtype)
                # GM [K, N] -> L1 [N, K] transpose at the documented GM->L1 stage, so
                # the L1 GEMM call is the NT form (transpose_A=False, transpose_B=True).
                b_l1 = T.alloc_l1((block_n, block_k), dtype)
                res = T.alloc_l0c((block_m, block_n), "float32")
                T.annotate_buffer_versions({a_l1: num_stages, b_l1: num_stages})
                sched.init(bx)
                while sched.valid():
                    batch_idx = sched.batch()
                    m_idx, n_idx, actual_m, actual_n = sched.tile()
                    for k_block_idx in T.Pipelined(num_k_blocks, num_stages=num_stages):
                        k_idx = k_block_idx * block_k
                        T.copy(
                            A[batch_idx, m_idx : m_idx + actual_m, k_idx : k_idx + block_k],
                            a_l1[0:actual_m, 0:block_k],
                        )
                        T.copy(
                            B[batch_idx, k_idx : k_idx + block_k, n_idx : n_idx + actual_n],
                            b_l1[0:actual_n, 0:block_k],
                            transpose=True,
                        )
                        # One GEMM per outer iteration over the full static L1 regions:
                        # the L1 lowering requires static M/N/K and owns the internal
                        # sub-K two-slot L0 loop. Do not replace it with a manual
                        # two-substep loop, and keep clear_accum a comparison
                        # expression -- round 2's trace-time if statement corrupted the
                        # enclosing L0 buffer-version analysis.
                        T.gemm(
                            a_l1,
                            b_l1,
                            res,
                            transpose_B=True,
                            clear_accum=(k_block_idx == 0),
                        )
                    T.copy(
                        res[0:actual_m, 0:actual_n],
                        D[batch_idx, m_idx : m_idx + actual_m, n_idx : n_idx + actual_n],
                    )
                    sched.next_block()

        return tilelang.compile(main, target="ascend", out_idx=-1)

    # Measured same-shape dispatch entries only: each key is exactly one workload
    # that has its own real measurement, and no entry is inferred from another
    # shape or dtype. Every other call, including all non-128-aligned M/N/K tails,
    # keeps the untouched L0 body above. This table is metadata-only: no input
    # value, benchmark state or module-global mutation participates.
    #
    # (B, M, N, K, dtype) -> (block_m, block_n, block_k, num_stages, num_blocks).
    # 001 measured MTE2 ~99.7% busy for both paths.  A 256x128 output tile
    # reduces the model A+B tile GM bytes per output by 25% versus 128x128
    # (it does not halve them), and its C side lowers TILE_K_SUB to 64.  The
    # finite screen below picked one measured rectangle per exact workload;
    # search/profile scores are not formal M4 results.
    _ALIGNED_L1_SELECTIONS = {
        # moe-prefill bf16: 0.479708 -> 0.378905 ms short
        # square-1k float16: 0.062241 -> 0.054295 ms short
        # square-1k bfloat16: 0.061389 -> 0.054969 ms short
        # square-2k float16: 0.408320 -> 0.330147 ms short
        # square-2k bfloat16: 0.409202 -> 0.329875 ms short
        # square-4k bfloat16: 1.538481 -> 1.209041 ms short
        (128, 512, 512, 2048, "bfloat16"): (256, 128, 256, 2, 36),
        (8, 1024, 1024, 1024, "float16"): (256, 128, 256, 2, 36),
        (8, 1024, 1024, 1024, "bfloat16"): (256, 128, 256, 2, 36),
        (8, 2048, 2048, 2048, "float16"): (256, 128, 256, 2, 36),
        (8, 2048, 2048, 2048, "bfloat16"): (256, 128, 256, 2, 36),
        (4, 4096, 4096, 4096, "bfloat16"): (256, 128, 256, 2, 36),
    }

    class AscendBmmKernel(Kernel, BmmFwdInterface):
        """Batched dense GEMM on Ascend NPUs: ``d[i] = a[i] @ b[i]`` per batch item.

        The project's first Cube/matmul kernel (batch 1 was 100% Vector/AIV elementwise;
        see ``docs/ascend950/dev/BmmFwdOp/round_1_schedule.md`` §5). A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``BmmKernel``/``BmmPersistentKernel`` family, which imports ``tilelang.language``
        (the CUDA dialect facade).
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: BmmCall) -> Optional[str]:
            """Refuse a K not a multiple of 16, matching the in-tree CUDA ``BmmKernel``'s
            own contract (``tileops/kernels/gemm/bmm.py``'s ``_k_refusal``) and
            ``BmmFwdOp``'s class docstring ("The in-tree kernels need K to be a multiple of
            16 and refuse other calls when built").

            Not just a convention match: this op's K-alignment table (schedule §4.1) states
            16-bit-dtype MAD K alignment is 16 elements, so an unaligned K is also a real
            hardware-contract mismatch for this kernel's L0 tiles, not merely a
            CUDA-kernel-specific restriction. ``test_bmm_k_not_multiple_of_16_raises``
            (``tests/ops/test_bmm.py``) exercises this without pinning a device/kernel, so
            whichever kernel ``BmmFwdOp`` dispatches to on a given device must uphold it.
            """
            return None if call.k % 16 == 0 else f"requires k a multiple of 16, got k={call.k}"

        @classmethod
        def entry_for(cls, call: BmmCall) -> Entry:
            return call, lambda: cls(call.batch, call.m, call.n, call.k, call.dtype)

        def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
            super().__init__()
            self.batch = batch
            self.m = m
            self.n = n
            self.k = k
            self.dtype = dtype
            dtype_name = _dtype_str(dtype)
            selection = _ALIGNED_L1_SELECTIONS.get((batch, m, n, k, dtype_name))
            if selection is None:
                self._compiled_fn = _make_bmm_kernel(batch, m, n, k, dtype_name)
            else:
                block_m, block_n, block_k, num_stages, num_blocks = selection
                self._compiled_fn = _make_bmm_aligned_l1_kernel(
                    batch, m, n, k, dtype_name,
                    block_m, block_n, block_k, num_stages, num_blocks,
                )

        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return self._compiled_fn(a, b)

    tileops.backend.register_kernel_type("BmmFwdOp", "ascend", AscendBmmKernel)

except ImportError:
    pass
