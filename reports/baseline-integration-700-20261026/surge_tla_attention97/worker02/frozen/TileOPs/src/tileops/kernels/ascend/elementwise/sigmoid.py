"""Ascend NPU Sigmoid forward kernel: ``devices={"npu"}``, outside the CUDA kernel hierarchy.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``SigmoidFwdOp``; see
``docs/ascend950/dev/SigmoidFwdOp/round_1_schedule.md`` for the tiling/schedule design this file
implements. Registers itself under the ``"ascend"`` dispatch key
(``tileops.backend.register_kernel_type``), so importing this module is enough to make
``SigmoidFwdOp`` route NPU tensors here -- no change to the existing CUDA ``SigmoidFwdKernel`` or
``SigmoidFwdOp``.

This is ``AscendSiluKernel`` (``tileops/kernels/ascend/elementwise/silu.py``) with its numerator
changed from ``wide`` to ``one``: ``sigmoid(x) = x * silu(x) / x = 1 / (1 + exp(-x))`` is
``_silu_op``'s own ``1 / (1 + exp(-x))`` sub-expression. The tiling scaffolding
(``_tile_elems_for_dtype``, ``_choose_num_blocks``, ``_build_single_block``,
``_build_decode_multicore``, ``_build_tiled``, the two-kernel tail-dispatch runner) is ported
verbatim from ``silu.py`` -- round_1_schedule.md SS0/SS3/SS4 found this op's 5 manifest cases are
even larger/more power-of-two-friendly than ``silu.py``'s (no decode-sized manifest case at all,
every case divides its tile size with zero tail), so the only lever actually exercised by M4 is
``num_blocks=64``, identical to ``silu.py``'s/``gelu.py``'s already-validated configuration. The
decode-multicore/tail machinery nonetheless has to ship unmodified because
``tests/ops/test_activation.py`` (M3, shared with ``TanhFwdOp``, not edited here) has fixtures at
4096/65536/262148+4 elements that land on exactly those branches (round_1_schedule.md SS4).

Deliberately **does not** port ``SigmoidFwdKernel.op_func``'s CUDA-only
``exp2(-x*log2e)``/``full_range_fdiv``/``refined_reciprocal`` trio
(``tileops/kernels/elementwise/activations.py:129-153``): that combination exists purely to
compensate for CUDA's approximate ``__fdividef`` instruction and round the result to
``torch.sigmoid``'s (CUDA) bit pattern at every 16-bit input -- Ascend has no equivalent
approximate-divide hardware instruction, and ``exp``/``exp2`` are both already-confirmed Ascend
intrinsics (``silu.py``'s ``T.exp`` direct translation already shipped). A plain fp32
``one / den`` is the correct translation of this op's math on this hardware; round_1_schedule.md
SS2/SS5.1 is explicit that this does **not** by itself prove the fp16/bf16 result is bit-for-bit
identical to ``torch_npu``'s own on-device ``torch.sigmoid`` (the standard this op's M3 suite
actually requires, via ``tests/ops/test_activation.py::test_16bit_sigmoid_matches_torch_bit_for_bit``)
-- see this op's ``round_1_result.md`` for what M3 measured.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports are guarded by one ``try/except ImportError``
(``_ASCEND_IMPORT_ERROR``), and the one real NPU-touching entry point
(``AscendSigmoidKernel.__init__``) re-raises it lazily, mirroring ``silu.py``/``gelu.py`` exactly
(``docs/ascend950/dev_process.md`` SS1).

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: ``docs/ascend950/dev_process.md`` SS1 explicitly asks every op's worker to self-
register from its own file and *not* build a shared aggregator file under ``kernels/ascend/``,
since several ops' workers (this batch: ``TanhFwdOp`` concurrently) write into this same new
directory at once and a shared ``__init__.py`` would be a file they all contend for.
``kernels/ascend`` and ``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420)
namespace packages, same precedent as every sibling file already here (``silu.py``, ``gelu.py``,
``relu.py``, ``tanh.py``).
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.carver.arch.driver import get_num_vector_cores

    _ASCEND_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # pragma: no cover - exercised only off-Ascend
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    get_num_vector_cores = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendSigmoidKernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from ``silu.py``'s own
# ``_UB_BUDGET_BYTES`` (round_1_schedule.md SS0.3/SS3): that value was confirmed on real 950
# hardware (M3 found a single hardcoded tile sized for fp16/bf16 overflows float32's dynamic
# shared-memory limit, which is how ``_tile_elems_for_dtype`` below ended up itemsize-aware).
# Not re-derived or re-verified here; this op's UB footprint (2 buffers, same two tile sizes)
# is identical in shape to ``silu.py``'s, so the same budget applies unchanged.
_UB_BUDGET_BYTES = 216 * 1024
# silu.py's round 1 sweep found 3 stages beat 2 at tile=16384/blocks=64, reused here per
# round_1_schedule.md SS0.3. Round 1 M4 measured all 3 16M cases ~12-14% over target while
# both 256M cases passed comfortably; an on-device A/B (num_stages=2 vs 3, this op's own 5
# cases, official protocol) found 3 still wins outright for 4 of 5 cases (worse at 2 for
# 16M-fp16/bf16 and both 256M cases; only 16M-fp32 preferred 2, by ~5%) -- see round_1_result.md
# SS4 for the full table. Kept at 3; the 16M shortfall has a different root cause (see
# round_1_result.md's diagnosis section), not a pipeline-stage-count problem.
_NUM_STAGES = 3
# Fallback vector-core count when the live query below is unavailable. Unlike ``silu.py``'s
# conservative ``_FALLBACK_VECTOR_CORES = 48`` (written before this SKU's real core count was
# confirmed), round_1_schedule.md SS0.3 directs using 72 here: ``get_num_vector_cores`` has
# since been confirmed (``ReluFwdOp``/``TanhFwdOp`` round 1/2, dev_process.md SS7.5) to report
# ``vector_core_num=72`` on this SKU (device ``Ascend950DT_9592``), so 72 is now a verified
# fact, not an unvalidated guess, and the fallback should say so.
_FALLBACK_VECTOR_CORES = 72
# Thread count the SimtVF region launches with. Reused unchanged from ``silu.py``: 2048 is a
# hard Ascend SIMT ceiling confirmed by M4 tuning on real hardware (``__launch_bounds__`` above
# 2048 fails to compile), not just a value copied from an example. Capped to the region's own
# element count for the decode/tail paths, which can be smaller than 2048.
_MAX_THREADS = 2048


def _sigmoid_op(x):
    """``1 / (1 + exp(-x))``, computed in fp32.

    ``silu.py``'s ``_silu_op`` is ``wide / (one + T.exp(-wide))``, i.e. ``x * sigmoid(x)``
    algebraically expanded; this is that same fp32 upcast, that same direct (not ``exp2``)
    ``T.exp`` call, and that same cast-back-to-storage-dtype at the boundary, with the
    numerator changed from ``wide`` to ``one`` -- the ``sigmoid(x)`` sub-expression ``silu``
    multiplies by ``x``. See this module's docstring and round_1_schedule.md SS2/SS5.1 for why
    this intentionally does not reach for ``SigmoidFwdKernel.op_func``'s CUDA-only
    ``exp2``/``full_range_fdiv``/``refined_reciprocal`` divide-compensation trio.
    """
    one = T.cast(1.0, "float32")
    wide = T.cast(x, "float32")
    return one / (one + T.exp(-wide))


def _num_vector_cores() -> int:
    """Best-effort AIV (vector) core count of the current NPU device.

    Queries the regulated ``tilelang.carver.arch.driver.get_num_vector_cores`` API
    (round_1_schedule.md SS0.3; confirmed by ``ReluFwdOp``/``TanhFwdOp`` to report 72 on this
    SKU), falling back to the literal ``_FALLBACK_VECTOR_CORES`` only if the live query raises
    or returns a falsy value. Only changes how many persistent blocks the multi-tile path
    launches with; a wrong or unavailable count degrades the launch geometry, never the result
    (``_choose_num_blocks`` always returns a divisor, so every block still runs the same exact
    iteration count). Deliberately not evaluated at import time -- it queries the real device,
    so this is only called lazily, the first time a kernel is actually built (always under an
    ``npu-guard`` reservation already).
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return int(queried)


def _tile_elems_for_dtype(dtype_str: str) -> int:
    """The largest power-of-two element count keeping the multi-tile path's UB use in budget.

    2 buffers (x_tile, y_tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
    ``_UB_BUDGET_BYTES``; a power of two keeps the resulting iteration/block-count arithmetic
    in ``_choose_num_blocks`` simple and gives every known benchmark ``n_total`` an exact or
    near-exact tiling. Ported unchanged from ``silu.py``: for fp16/bf16 (itemsize 2) this
    returns 16384 (192 KiB used); for float32 (itemsize 4) it returns 8192 (also 192 KiB used)
    -- round_1_schedule.md SS3 re-derived and confirmed both numbers match ``silu.py``'s.
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    tile = 1
    while tile * 2 <= max_elems:
        tile *= 2
    return tile


def _choose_num_blocks(num_full_tiles: int, vector_cores: int) -> int:
    """The largest divisor of ``num_full_tiles`` that does not exceed ``vector_cores``.

    Guarantees ``num_full_tiles % result == 0``, so every persistent block runs the exact same
    number of iterations -- no per-block tail inside the main loop, regardless of whether
    ``num_full_tiles`` happens to be round. Ported unchanged from ``silu.py``. For this op's 5
    manifest cases, ``num_full_tiles`` is always a power of two (2^10, 2^11, or 2^14) and
    ``vector_cores`` is 72 (not a power of two), so the search lands on 64 for every one of
    them (round_1_schedule.md SS3 derivation table) -- the same value ``silu.py``/``gelu.py``
    already validated on hardware. 1 always qualifies, so this never fails to return a valid
    schedule, even for a prime ``num_full_tiles`` larger than every core count.
    """
    cap = max(1, min(num_full_tiles, vector_cores))
    for candidate in range(cap, 0, -1):
        if num_full_tiles % candidate == 0:
            return candidate
    return 1  # unreachable: candidate == 1 always divides num_full_tiles


if T is not None:

    @functools.lru_cache(maxsize=None)
    def _build_single_block(extent: int, dtype_str: str):
        """One block, one untiled GM->UB->compute->UB->GM pass over ``extent`` elements.

        Used both for a decode-sized call too small to pipeline and, reused unchanged, as the
        tail epilogue of the tiled path below when ``n_total`` is not an exact multiple of the
        dtype's tile size. Ported unchanged from ``silu.py``; none of this op's 5 manifest
        cases reaches this path (round_1_schedule.md SS3/SS4), but
        ``tests/ops/test_activation.py::test_sigmoid_partial_last_block`` (262148 = 2**18 + 4
        float32 elements) exercises it as the tail kernel for the +4 remainder.
        """

        @T.prim_func
        def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
            with T.Kernel(1):
                x_tile = T.alloc_shared((extent,), dtype_str)
                y_tile = T.alloc_shared((extent,), dtype_str)
                T.copy(X, x_tile)
                with T.SimtVF(threads=min(extent, _MAX_THREADS)):
                    for i in T.Parallel(extent):
                        y_tile[i] = _sigmoid_op(x_tile[i])
                T.copy(y_tile, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_decode_multicore(extent: int, dtype_str: str, num_cores: int):
        """Grid-of-``num_cores`` sigmoid kernel for a small ``extent`` that divides evenly by
        ``num_cores``.

        Ported unchanged from ``silu.py``'s ``_build_decode_multicore`` (itself porting
        ``gelu.py``'s proven fix for the decode case: spreading a small element count across a
        multi-core grid rather than running the whole extent on one of 72 idle-elsewhere
        cores). This op's manifest has no decode-sized case (round_1_schedule.md SS1/SS4), but
        ``tests/ops/test_activation.py::test_sigmoid_edge`` and
        ``test_sigmoid_input_off_a_vector_boundary`` (both ``n_total=4096``, float32) land here
        since 4096 <= ``_tile_elems_for_dtype("float32")`` (8192). Every core gets exactly one
        statically-sized ``block = extent // num_cores``-element tile -- no re-looping, hence a
        plain ``T.Kernel(num_cores) as core_id:`` body rather than ``T.Persistent``.
        """
        assert extent % num_cores == 0, (
            f"_build_decode_multicore requires extent={extent} to divide evenly by "
            f"num_cores={num_cores}"
        )
        block = extent // num_cores
        threads = min(block, _MAX_THREADS)

        @T.prim_func
        def main(X: T.Tensor((extent,), dtype_str), Y: T.Tensor((extent,), dtype_str)):
            with T.Kernel(num_cores) as core_id:
                x_tile = T.alloc_shared((block,), dtype_str)
                y_tile = T.alloc_shared((block,), dtype_str)
                begin = core_id * block
                T.copy(X[begin : begin + block], x_tile)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(block):
                        y_tile[i] = _sigmoid_op(x_tile[i])
                T.copy(y_tile, Y[begin : begin + block])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=None)
    def _build_tiled(main_n: int, dtype_str: str, num_blocks: int, tile_elems: int):
        """Persistent-block, pipelined main loop over exact multiples of ``tile_elems``.

        ``main_n`` is an exact multiple of ``num_blocks * tile_elems`` by construction (the
        caller, ``_build_sigmoid_kernel``, only ever passes the full-tile portion of ``n_total``
        and a ``num_blocks`` that divides its tile count), so no tail handling is needed inside
        this program itself. Ported unchanged from ``silu.py``, including its ``iterations``
        recomputed-inside-the-body workaround for TileLang's eager ``get_type_hints`` closure
        limitation. Serves all 5 manifest cases, plus
        ``tests/ops/test_activation.py::test_sigmoid`` (1,048,576 elements, all 3 dtypes) and
        the main 262144-element body of ``test_sigmoid_partial_last_block``.
        """

        @T.prim_func
        def main(X: T.Tensor((main_n,), dtype_str), Y: T.Tensor((main_n,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                iterations = main_n // (num_blocks * tile_elems)
                x_tile = T.alloc_shared((tile_elems,), dtype_str)
                y_tile = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_tile: _NUM_STAGES, y_tile: _NUM_STAGES})
                for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_tile)
                    with T.SimtVF(threads=min(tile_elems, _MAX_THREADS)):
                        for i in T.Parallel(tile_elems):
                            y_tile[i] = _sigmoid_op(x_tile[i])
                    T.copy(y_tile, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)


class _SigmoidRunner:
    """Dispatches one flat ``(n_total,)`` NPU tensor to one or two compiled Ascend programs.

    Two programs only when ``n_total`` exceeds the tile size and does not divide it evenly:
    the tiled/pipelined program covers every full tile, and a second single-block program (the
    same builder the small/decode case uses) covers the remainder. Ported unchanged from
    ``silu.py``'s ``_SiluRunner`` (plain Python dispatch around two already-compiled kernels,
    not a runtime branch inside either kernel); only the class name changed, to match this
    module's ``_build_sigmoid_kernel``/``_sigmoid_op`` naming.
    """

    __slots__ = ("main_kernel", "tail_kernel", "main_n")

    def __init__(self, main_kernel, tail_kernel, main_n: int):
        self.main_kernel = main_kernel
        self.tail_kernel = tail_kernel
        self.main_n = main_n

    def __call__(self, flat_input: torch.Tensor) -> torch.Tensor:
        if self.tail_kernel is None:
            return self.main_kernel(flat_input)
        tail_part = self.tail_kernel(flat_input[self.main_n :])
        if self.main_kernel is None:
            return tail_part
        main_part = self.main_kernel(flat_input[: self.main_n])
        return torch.cat((main_part, tail_part))


@functools.lru_cache(maxsize=None)
def _build_sigmoid_kernel(n_total: int, dtype_str: str) -> "_SigmoidRunner":
    """Build (or reuse, by ``(n_total, dtype_str)``) the Ascend sigmoid program.

    ``n_total <= tile_elems``: spread across ``_choose_num_blocks(n_total,
    _num_vector_cores())`` cores, one statically-sized tile per core, via
    ``_build_decode_multicore``. Falls back to ``_build_single_block`` only when the real core
    count's best divisor of ``n_total`` is 1 (e.g. a prime ``n_total`` larger than every core
    count).

    ``n_total > tile_elems``: a persistent-block main loop over ``tile_elems``-sized tiles,
    plus -- when ``n_total`` is not an exact multiple of ``tile_elems`` -- a second,
    differently-sized single-block program for the remainder (see ``_SigmoidRunner``). Every
    sub-builder is memoized with ``functools.lru_cache``, and ``n_total``/``dtype_str`` are
    Python-level facts fixed per compiled kernel instance (one per distinct
    ``AscendSigmoidKernel(n_total, dtype)``), so every ``n_total`` gets a *correct* schedule,
    not one that assumes even division. Ported unchanged from ``silu.py``'s
    ``_build_silu_kernel``, substituting ``_sigmoid_op``/``_build_decode_multicore``/
    ``_build_tiled``/``_build_single_block``'s sigmoid-flavored builders defined above.
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")

    tile_elems = _tile_elems_for_dtype(dtype_str)
    if n_total <= tile_elems:
        num_cores = _choose_num_blocks(n_total, _num_vector_cores())
        if num_cores > 1:
            return _SigmoidRunner(
                _build_decode_multicore(n_total, dtype_str, num_cores), None, n_total
            )
        return _SigmoidRunner(_build_single_block(n_total, dtype_str), None, n_total)

    num_full_tiles = n_total // tile_elems
    tail = n_total - num_full_tiles * tile_elems
    main_n = num_full_tiles * tile_elems
    # round_1_schedule.md SS3: this op's 5 manifest cases all land on num_blocks=64 here,
    # identical to silu.py's/gelu.py's already hardware-validated prefill configuration (same
    # cost shape, same tile sizes, same vector-core count) -- not re-swept this round.
    num_blocks = _choose_num_blocks(num_full_tiles, _num_vector_cores())
    main_kernel = _build_tiled(main_n, dtype_str, num_blocks, tile_elems)
    tail_kernel = _build_single_block(tail, dtype_str) if tail else None
    return _SigmoidRunner(main_kernel, tail_kernel, main_n)


class AscendSigmoidKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend NPU sigmoid: ``1 / (1 + exp(-x))`` in fp32, cast back to the storage dtype.

    See ``docs/ascend950/dev/SigmoidFwdOp/round_1_schedule.md`` for the tiling design and this
    op's own ``round_1_result.md`` for what the bit-exactness (M3) and performance (M4)
    verification actually found on hardware. ``devices`` is NPU only; ``supported_archs`` is
    left ``None`` -- unlike the CUDA ``UnaryKernel``/``FloatUnaryKernel`` hierarchy this class
    does not inherit, there is no Ascend SoC-version gate here to replace the CUDA SM-version
    one (``Kernel._check_arch`` skips the check entirely when ``supported_archs is None``).

    Tolerance for the shared float32 case is inherited from the shared elementwise test
    tolerance table (not re-derived here); the fp16/bf16 cases additionally face
    ``tests/ops/test_activation.py::test_16bit_sigmoid_matches_torch_bit_for_bit``, which
    requires an exact bit-for-bit match (``torch.equal`` on the integer view, NaN payloads
    included) against ``torch_npu``'s own on-device ``torch.sigmoid`` at all 65536 16-bit
    codes -- see this module's docstring for why this kernel does not attempt CUDA's
    divide-compensation tricks to chase that bound, and this op's ``round_1_result.md`` for the
    measured outcome.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendSigmoidKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.kernel = _build_sigmoid_kernel(n_total, self.dtype_str)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed. May be a view whose storage offset is not
                16-byte aligned (round_1_schedule.md SS5.2); ``T.copy``'s GM-side DMA reads by
                byte offset, with no alignment requirement demonstrated to apply there.

        Returns:
            A new contiguous tensor on the same device, shaped and typed as ``input``.
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self.kernel(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("SigmoidFwdOp", "ascend", AscendSigmoidKernel)
except ImportError:
    pass
