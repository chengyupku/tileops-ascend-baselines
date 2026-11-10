"""Ascend NPU dropout forward kernel: inverted dropout via the Ascend Philox4x32-10 RNG.

New Ascend-only implementation of ``DropoutFwdOp``, registered under the dispatch key
``"ascend"`` (``docs/ascend950/dev_process.md`` §1) alongside the existing in-tree CUDA key
``"dropout"`` (``tileops.kernels.elementwise.dropout.DropoutKernel``). This module only *adds*
an implementation -- it does not import, edit, or otherwise touch
``tileops/kernels/elementwise/dropout.py`` or ``tileops/ops/elementwise/dropout.py``. It reuses
``DropoutFwdOp``'s existing ``DropoutCall``/``DropoutFwdInterface`` contract unmodified: element
``i`` becomes ``x[i] / (1 - p)`` where its draw keeps it and ``0`` otherwise, over a flat
``(call.count,)`` run, with ``call.seed`` fixing the mask for deterministic replay. The op's own
``forward()`` already short-circuits ``p in (0, 1)`` and ``training=False`` to
``.clone()``/``zeros_like`` before ever calling ``kernel_for``, so this kernel only ever sees
``0 < p < 1``, ``training=True``.

Design: ``docs/ascend950/dev/DropoutFwdOp/round_1_schedule.md`` (Planner, round 1), which
reuses ``GeluFwdOp``/``ReluFwdOp``'s already-measured persistent-grid pattern
(``kernels/ascend/elementwise/{gelu,relu}.py``) for the op's three prefill-sized manifest cases
(no decode case for this op), plus one genuinely new piece: per-``(iteration, core, lane)`` RNG
stream derivation (§3.3 below).

Correctness bar (``workloads/elementwise.py``'s ``ElementwiseWorkload.verification`` override for
``DropoutFwdOp``, exercised by ``tests/ops/test_dropout.py``) is statistical, not bit-exact
against CUDA's cuRAND stream: exact ``x / (1 - p)`` scaling on every kept element, a drop count
within six sigma of ``n * p``, and same-seed-same-backend determinism. This leaves the exact
``seq``/``off`` convention a free design choice (schedule §1.2) -- the one below does not aim to
reproduce CUDA's ``blockIdx*blockDim+threadIdx`` derivation.

Importing this module on a process with no Ascend toolchain installed is a no-op: the
``tilelang.ascend`` import is wrapped in ``try/except ImportError``, so no class is defined and
no registration happens.

## §3.3 RNG stream derivation (schedule's new-territory design)

Every existing elementwise pilot re-enters a *fresh* ``T.SimtVF`` scope each ``T.Pipelined``
iteration (DMA always lives outside any ``SimtVF``), so ``T.rng_init`` is re-issued every
iteration. Ascend's default ``seq`` derivation (lane id + core index * lane count, see
``tilelang/ascend/language/random.py``) does not vary with the iteration index on its own --
reusing it unmodified would silently replay the identical mask on every pipeline iteration. The
fix, used by both builders below: fold the iteration ``step`` into an explicit ``seq`` ourselves,
exactly the way ``begin`` already folds ``step``/``bx``/``tile`` into a global element offset one
line above it -- ``seq = (step * num_blocks + bx) * threads + tx`` gives every
``(step, bx, tx)`` triple a distinct Philox subsequence (subsequences are independent by
construction regardless of ``off``), so there is no collision risk across iterations or cores.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.dropout import DropoutCall, DropoutFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # no Ascend-aware tilelang in this process: safe no-op import
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendDropoutKernel"]

# Persistent-grid geometry (schedule §3.1), carried over from ReluFwdOp round 2's calibrated
# value: `_floor_pow2(72)` real vector cores on this SKU (`get_num_vector_cores`), confirmed by
# a corrected, no-cache-flush M4 sweep to beat both 32 and 128 for bandwidth-bound prefill
# elementwise work. Not re-queried live here: round 1 of every elementwise pilot so far has
# used the literal 64 once the live query confirmed it, rather than paying a per-build device
# round trip (`relu.py`'s `_calibrated_num_blocks` is the one exception, kept lazy specifically
# because its decode path shares the constant across a wider set of shapes; this op has no
# decode case, so there is no such pressure here).
_NUM_BLOCKS = 64
_NUM_STAGES = 2

# Per-dtype UB tile size, in elements: fp16/bf16 get 8192 (16 KiB/buffer), float32 is halved to
# 4096 to hold the same 16 KiB/buffer footprint at twice the element width -- BitwiseNotFwdOp
# round 1's "same byte footprint, half the width for a wider dtype" precedent (that op halved
# int64's tile relative to int32 for the same reason), applied here for fp32 vs fp16/bf16 since
# this is the first batch-2 op with a large persistent-tile float32 case.
_TILE_BY_DTYPE = {"float16": 8192, "bfloat16": 8192, "float32": 4096}

# Per-dtype SIMT thread count for the persistent path, in place of the naive `min(tile, 2048)`
# every other batch-1/2 elementwise kernel uses: schedule §3.4 measured 0 vectorized
# `philox_rand_uniform2/4` calls for this op's body in every dtype (the `if_then_else`/scale
# arithmetic between the draw and the store blocks whatever pattern the loop-vectorizer
# matches), so each thread's draws fall back to scalar `philox_rand_uniform`, which still
# buffers 4 words per underlying Philox10 block (`philox_rng.h`). At `min(tile, 2048)` threads,
# fp16/bf16 (tile=8192) get 4 elements/thread -- a full buffer's worth of entropy used exactly
# once per `rng_init` -- while float32 (tile=4096) gets only 2 elements/thread, wasting half of
# every generated buffer and paying a `philox_init` per thread per iteration for half the
# payoff. M4 round 1 confirmed this mattered in practice: at `threads=2048` float32 measured
# 64.62us against its 80% bar of 50.41us (`1.25 * best_baseline_latency`, 40.33us baseline) --
# 1.60x over, a real miss, not a rounding error. Halving float32's thread count to 1024 (tile
# unchanged at 4096) restores 4 elements/thread -- full buffer reuse, half as many `rng_init`
# calls for the same data volume -- re-measured under the official (no-cache-flush) M4 protocol
# per `dev_process.md` §7.5's `msprof_detail` caveat, not just a cache-flushed `do_bench` sweep:
# 34.48us, comfortably inside the 50.41us bar. See round_1_result.md for both numbers.
_THREADS_BY_DTYPE = {"float16": 2048, "bfloat16": 2048, "float32": 1024}

if T is not None:

    @T.macro
    def _dropout_iter_body(
        x_ub,
        y_ub,
        n: int,
        threads: int,
        seed: int,
        p: float,
        scale: float,
        step,
        num_blocks: int,
        bx,
    ) -> None:
        """One tile's worth of inverted dropout, with a step-folded RNG stream (module
        docstring §3.3).

        ``step``/``bx`` are TileLang loop/kernel-binding expressions (a ``T.Pipelined`` index
        and a ``T.Kernel(...) as bx`` core index) for the persistent-grid builder, or plain
        Python ``int`` constants for the single-block fallback's tail chunk -- both are valid
        operands inside the ``seq`` expression below, exactly as ``begin`` already mixes the
        same two kinds of value one line above every caller site.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"; a plain function's ``for i in
        T.Parallel(n):`` is not rewritten by the eager builder and fails with ``TypeError:
        'ForFrame' object is not iterable``).
        """
        with T.SimtVF(threads=threads):
            tx = T.get_thread_binding()
            # Globally unique per (iteration, core, lane): see module docstring §3.3.
            seq = (step * num_blocks + bx) * threads + tx
            T.rng_init(seed, seq=seq, off=0)
            for i in T.Parallel(n):
                r = T.rng_rand_float(dist="uniform")
                keep = r >= T.cast(p, "float32")
                y_ub[i] = T.if_then_else(
                    keep,
                    T.cast(x_ub[i] * T.cast(scale, x_ub.dtype), x_ub.dtype),
                    T.cast(0.0, x_ub.dtype),
                )

    @functools.lru_cache(maxsize=64)
    def _build_persistent(
        n_total: int, dtype_str: str, p: float, seed: int, num_blocks: int, tile: int, num_stages: int
    ):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule §3.1/§3.3).

        Serves this op's three manifest cases, all of which divide ``num_blocks * tile``
        exactly (schedule §3.1: "no tail-handling path is needed this round" for the
        *manifest* cases specifically). ``_make_dropout_kernel`` only reaches this builder for
        an ``n_total`` that divides evenly; every other shape (this op's own
        ``tests/ops/test_dropout.py`` fixtures use 1,000,000- and 4,000,000-element shapes that
        do not) routes to ``_build_generic_single_block`` instead.
        """
        scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_build_persistent requires n_total ({n_total}) to divide num_blocks*tile "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span
        # Per-dtype thread count (not the naive `min(tile, 2048)`): see `_THREADS_BY_DTYPE`'s
        # comment for why float32 needs fewer, wider-per-thread lanes to reuse the Philox
        # RNG's 4-word buffer fully.
        threads = _THREADS_BY_DTYPE[dtype_str]

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype_str)
                y_ub = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _dropout_iter_body(x_ub, y_ub, tile, threads, seed, p, scale, step, num_blocks, bx)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _build_generic_single_block(n_total: int, dtype_str: str, p: float, seed: int, tile: int):
        """Single-core fallback for an ``n_total`` the persistent tiling does not evenly divide.

        Needed because this op's own correctness-test fixtures (``test_dropout.py``: 1M- and
        4M-element shapes) are not the manifest's exact prefill sizes, mirroring
        ``ReluFwdOp``'s ``_make_relu_generic_single_block`` (round_1_schedule.md risk 1 there):
        a main ``T.Pipelined`` loop over full tile-sized chunks, plus one statically-sized tail
        chunk, all on one AI core. Each chunk gets its own ``_dropout_iter_body`` call, so the
        tail chunk still draws a fresh, step-indexed RNG stream rather than reusing the main
        loop's last iteration's mask. Not performance-tuned (single core): this op's three
        M4-benchmarked manifest cases never reach this function; it exists purely so M3's
        differently-shaped correctness fixtures see a correct (if unoptimized) Ascend path
        instead of a hard ``NotImplementedError``.
        """
        scale = 1.0 / (1.0 - p) if p < 1.0 else 0.0
        full_tiles, remainder = divmod(n_total, tile)
        threads_main = min(tile, 2048)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype_str)
                    y_ub = T.alloc_shared((tile,), dtype_str)
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _dropout_iter_body(
                            x_ub, y_ub, tile, threads_main, seed, p, scale, step, 1, 0
                        )
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype_str)
                    yt_ub = T.alloc_shared((remainder,), dtype_str)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    threads_tail = min(remainder, 2048)
                    # step=full_tiles (one past the main loop's last index): a distinct,
                    # never-reused step value, so the tail's RNG stream cannot collide with
                    # the main loop's (module docstring §3.3).
                    _dropout_iter_body(
                        xt_ub, yt_ub, remainder, threads_tail, seed, p, scale, full_tiles, 1, 0
                    )
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_dropout_kernel(n_total: int, dtype_str: str, p: float, seed: int):
        """Pick the persistent-grid or single-block-fallback tiling for ``n_total`` elements.

        Cached on ``(n_total, dtype_str, p, seed)``, not just ``(n_total, dtype_str)`` like
        ``GeluFwdOp``/``ReluFwdOp``'s builders: ``p`` and ``seed`` are baked into the compiled
        program as Python-level constants (schedule §3.3), the same caching granularity the
        in-tree CUDA ``_make_dropout_kernel`` (``kernels/elementwise/dropout.py``) already uses.
        """
        tile = _TILE_BY_DTYPE[dtype_str]
        block_span = _NUM_BLOCKS * tile
        if n_total % block_span == 0:
            return _build_persistent(n_total, dtype_str, p, seed, _NUM_BLOCKS, tile, _NUM_STAGES)
        return _build_generic_single_block(n_total, dtype_str, p, seed, tile)


class AscendDropoutKernel(Kernel, DropoutFwdInterface):
    """Ascend NPU inverted dropout: ``y = where(rand() >= p, x / (1 - p), 0)``.

    Uses the Ascend dialect's hardware-native Philox4x32-10 RNG (``T.rng_init`` /
    ``T.rng_rand_float``, see the module docstring's §3.3) in place of CUDA's cuRAND, with its
    own per-``(iteration, core, lane)`` stream derivation; this is *not* bit-exact with the
    in-tree CUDA ``DropoutKernel``'s mask for the same seed (no such requirement exists -- see
    the module docstring's "Correctness bar" paragraph), only equally deterministic on repeat
    calls to the same compiled Ascend kernel. Registered under the ``"ascend"`` dispatch key,
    not ``"dropout"`` (already taken by the CUDA kernel on the same interface); selected
    automatically for NPU-device calls because ``devices = frozenset({"npu"})`` makes the CUDA
    key ``unavailable()`` there.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def entry_for(cls, call: DropoutCall) -> Entry:
        return call, lambda: cls(call.count, call.dtype, call.p, call.seed)

    def __init__(
        self, n_total: int, dtype: torch.dtype, p: float, seed: int, config: Optional[dict] = None
    ) -> None:
        if T is None:
            raise ImportError(
                "AscendDropoutKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.p = p
        self.seed = seed
        self._compiled_fn = _make_dropout_kernel(n_total, self.dtype_to_str(dtype), p, seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inverted dropout on the flat ``(call.count,)`` input ``x``; see
        ``DropoutFwdInterface.forward`` for the exact contract."""
        if x.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on {x.device}. "
                "Another target's backend serves other devices."
            )
        return self._compiled_fn(x.reshape(-1))


if T is not None:
    register_kernel_type("DropoutFwdOp", "ascend", AscendDropoutKernel)
