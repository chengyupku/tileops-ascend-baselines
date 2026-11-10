"""Ascend NPU IsInf forward kernel: ``y = isinf(x)``, bool output, via the
``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for
``IsinfFwdOp``; see ``docs/ascend950/dev/IsinfFwdOp/round_1_schedule.md`` for the full
design (manifest, interface, UB math, compile probes). Registers an additional
implementation of ``IsinfFwdOp``'s own ``UnaryPredicateFwdInterface`` under the
``"ascend"`` dispatch key (``tileops.backend.register_kernel_type``), beside the
in-tree CUDA ``IsinfFwdKernel``/``AlwaysFalseFwdKernel`` pair
(``tileops.kernels.elementwise.comparison``). No existing file is edited.

**Interface choice is load-bearing** (schedule doc §2): ``IsinfFwdOp``
(``tileops/ops/elementwise/comparison.py:99-106``) declares
``interfaces = {ELEMENTWISE: UnaryPredicateFwdInterface}``, a sibling class of the
generic ``UnaryElementwiseFwdInterface`` most unary ops use, with an identical
abstract ``forward(self, input)`` shape. Inheriting the wrong one here would not be
caught by any signature check, only by ``op_base.py``'s ``issubclass`` filter, which
would leave this kernel's key "unassigned" and raise ``ValueError: IsinfFwdOp keys
implement none of its kernel interfaces`` at ``IsinfFwdOp()`` construction, on every
device, every test. Hence ``AscendIsinfKernel(Kernel, UnaryPredicateFwdInterface)``.

**Manifest domain, re-confirmed directly from the live YAML rather than assumed from
a sibling's write-up**: ``src/tileops/manifest/spec/elementwise_unary_math.yaml:449``
-- ``IsinfFwdOp``'s ``forall`` is ``bool | uint8 | int8 | int16 | int32 | int64 |
float16 | bfloat16 | float32``, 9 dtypes total, byte-for-byte identical to
``IsnanFwdOp``'s own ``forall`` (``elementwise_unary_math.yaml:430``). Only the 3
float dtypes can represent +-Inf; the other 6 (5 int dtypes + bool) are answered
directly in ``forward()`` with ``torch.zeros(input.shape, dtype=torch.bool,
device=input.device)`` -- bit-identical to the CUDA ``AlwaysFalseFwdKernel.handler``
(``kernels/elementwise/comparison.py:210-217``), and deliberately the *same*
expression, not re-derived: ``torch.zeros`` never reads ``input``'s data, so there is
no ACLNN dtype-width coverage risk to verify the way there was for ``NegFwdOp``'s/
``SignFwdOp``'s int fallbacks. ``refusal()`` therefore accepts the full 9-dtype
domain -- narrowing it to floats would leave an ``npu`` + int/bool call with zero
eligible kernels (``AlwaysFalseFwdKernel`` is CUDA-only, refuses on device), failing
``test_predicate_non_float_constant`` on an npu run.

**The int8/bool-output vectorization hazard (schedule doc §5), re-verified directly
for this op's own expression before shipping, not just assumed from the schedule**:
a wide ``SimtVF`` scope (elems/thread > 1) that stores a 1-byte result (bool or int8)
while reading a 2-byte input (fp16/bf16) fails to compile: ``SimtVF cannot convert
vector type int8x2/int8x4 to an Ascend type``. Two things were checked fresh in this
round, beyond what the schedule had already probed:

1. The hazard reproduces identically whether the output buffer is declared ``"bool"``
   or ``"int8"`` -- both fail starting at exactly elems/thread=2 for a bare
   ``y_ub[i] = T.isinf(T.cast(x_ub[i], "float32"))`` (tile=4096, threads=2048, the same
   boundary the schedule's own ``int8``-only probe found). This matters because the
   sibling ``isnan.py`` in this same directory claims (its own module docstring) that
   its analogous bool-output assignment "compiles cleanly at 4 elements/thread"
   (tile=8192). That claim does not generalize to this op's own ``T.isinf`` expression
   -- confirmed directly, not assumed -- most likely because ``isinf`` lowers to a
   compound expression (``abs(x) == inf && !isnan(x)``, TVM ``op.cc:914-970``) rather
   than ``isnan``'s single compare, which may trigger the vectorizer's packing attempt
   differently. Not traced further into codegen; treated as an empirical fact. Worth
   flagging for whoever builds ``IsfiniteFwdOp`` (also composite for 16-bit inputs,
   ``!isinf(x) && !isnan(x)``, likely at least as exposed to this as ``isinf``).
2. The validated workaround (schedule doc §5.1) -- keep the DMA tile wide, but
   Python-unroll the compute+store body into independent 2048-element chunks, each
   its own ``T.SimtVF(threads=2048)`` scope -- compiles for all 3 float dtypes at every
   tile width tried (8192 through 32768) with a ``"bool"`` output buffer (not just
   ``"int8"``), so this file uses ``"bool"`` directly end to end (no ``int8`` storage /
   ``.view(torch.bool)`` boundary conversion needed) -- simpler than the schedule's
   originally-sketched ``int8``-storage design, functionally identical.

**Asymmetric UB/tile math** (schedule doc §6) gives the UB-budget-*maximal* tile: with
``itemsize_out=1`` (bool) and ``_UB_BUDGET_BYTES = 216 * 1024``, ``num_stages=2`` gives
``tile = floor_pow2(216*1024 // (2*(itemsize_in+1)))`` = 32768 for fp16/bf16 and 16384
for float32. **M4 found this UB-ceiling tile is not the latency-optimal one**, however
(round_1_result.md §4 has the full sweep): total ``T.SimtVF`` *scope-entry* count for a
fixed ``n_total`` is ``n_total // 2048`` regardless of the chosen DMA tile width (every
2048-element chunk needs its own scope either way), and this op's measured latency
tracks that per-scope-entry cost (empirically ~340-350ns/scope on this SKU, derived
from `measured_ms / (n_total // 2048 // num_blocks)`) far more than it tracks DMA tile
width -- widening the tile from 8192 to 32768 changed measured latency by under 10%.
The shipped tile (``_TILE_FOR_DTYPE`` below) is therefore the narrowest sweep point
tested (8192 for fp16/bf16, 16384 for float32, the latter because 16384 beat 8192 for
float32 specifically) rather than the UB-ceiling one -- a Pareto improvement over every
other tile combination measured, not merely "narrower and simpler". This chunking
workaround's fixed per-scope cost is also the primary reason this round's targets were
not met on any of the 5 cases; see round_1_result.md for the full diagnosis and a
round-2 direction (output bit-packing, avoiding the per-2048-element scope altogether).

All 5 benchmarked cases (2**24 or 2**28 elements) divide ``num_blocks(=64) * tile`` with
zero remainder -- no tail path needed for the benchmarked cases. A generic single-block
fallback (mirroring ``relu.py``/``isnan.py``) still exists below for the one
non-benchmarked shape this op's own test surface needs (the 1,048,576-element
``test_unary_bool_compile`` compile-smoke fixture), since that shape does not divide
``64 * tile`` evenly for either tile value this file uses.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a
CUDA-only process) is a no-op: the Ascend-specific imports are guarded by one
``try/except ImportError`` (``_ASCEND_IMPORT_ERROR``), re-raised lazily only when
``AscendIsinfKernel.__init__`` is actually reached, mirroring every sibling Ascend
kernel file in this directory.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports -- the
same deliberate omission every sibling file in this directory documents (several ops'
workers write into this same directory concurrently; a shared aggregator file would be
contended, ``docs/ascend950/dev_process.md`` §1).
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryPredicateFwdInterface
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

__all__ = ["AscendIsinfKernel"]

# The 3 float dtypes that can represent +-Inf: the only ones a real TileLang program is
# built for. Re-confirmed against the live manifest (module docstring) rather than
# assumed: 9 dtypes total for this op, same as IsnanFwdOp.
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# The other 6 manifest dtypes: no +-Inf representation in any of them, so forward()
# answers them directly with torch.zeros(..., dtype=torch.bool) -- no TileLang program,
# no ACLNN dispatch, bit-identical to AlwaysFalseFwdKernel.handler.
_INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_FALLBACK_DTYPES = _INT_DTYPES + (torch.bool,)

# Per-dtype DMA tile, chosen by M4 sweep (round_1_result.md §4), not by the UB-budget
# ceiling (module docstring: schedule doc §6's UB math permits 32768/16384, but measured
# latency is dominated by the fixed per-2048-element-chunk SimtVF scope-entry cost, which
# is ~tile-invariant -- these two values were the best of {8192, 16384, 32768} tried).
_TILE_FOR_DTYPE = {"float16": 8192, "bfloat16": 8192, "float32": 16384}
# Elements per independent SimtVF scope within one tile (module docstring: the
# int8/bool-output vectorization hazard's own safe boundary is elems/thread == 1, i.e.
# threads == chunk). Keeping the chunk fixed at the hazard's own safe width, rather than
# scaling it with tile, is what lets the DMA tile widen independently of this constraint.
_CHUNK = 2048
_NUM_STAGES = 2
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a
# falsy value. 64 is this SKU's own measured-optimal value (ReluFwdOp/CeilFwdOp/ExpFwdOp
# and every batch-3/4 unary op since), not an arbitrary guess.
_FALLBACK_VECTOR_CORES = 64
# T.SimtVF's hard 2048-thread ceiling (dev_process.md §7.5), confirmed project-wide.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks() -> int:
    """Persistent-grid block count for the tiled, bandwidth-bound path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0)
    and floors it to the nearest power of two, matching every sibling Ascend elementwise
    kernel's ``_calibrated_num_blocks``. Deliberately **not** evaluated at module import
    time -- the query touches the real device, so merely importing this module must stay
    a no-NPU-touch no-op. ``lru_cache`` makes it a one-time query per process either way.
    """
    try:
        queried = get_num_vector_cores(device_id=0)
    except Exception:
        queried = None
    if not queried:
        return _FALLBACK_VECTOR_CORES
    return _floor_pow2(int(queried))


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


if T is not None:

    @T.macro
    def _isinf_body(x_ub, y_ub, n: int) -> None:
        """``y_ub[:n] = isinf(x_ub[:n])`` as bool, computed in fp32.

        Python-unrolled into independent ``_CHUNK``-sized (2048-element)
        ``T.SimtVF(threads=_CHUNK)`` scopes, each individually at elems/thread=1 -- the
        validated workaround for the int8/bool-output vectorization hazard (module
        docstring, round_1_schedule.md §5.1): a *single* wide scope with elems/thread > 1
        fails to compile (``SimtVF cannot convert vector type int8x2/int8x4 to an Ascend
        type``) for a 2-byte (fp16/bf16) input paired with a 1-byte (bool) output. One
        code path for all 3 float dtypes, including float32 (which does not strictly need
        chunking, but using it uniformly avoids a dtype-conditional kernel-building
        branch and costs nothing -- the chunked float32 form compiles identically).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- an undecorated helper's ``for i in
        T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        under TileLang's eager builder.
        """
        full_chunks, remainder = divmod(n, _CHUNK)
        for c in range(full_chunks):
            base = c * _CHUNK
            with T.SimtVF(threads=_CHUNK):
                for i in T.Parallel(_CHUNK):
                    xf = T.Cast("float32", x_ub[base + i])
                    y_ub[base + i] = T.isinf(xf)
        if remainder:
            base = full_chunks * _CHUNK
            with T.SimtVF(threads=remainder):
                for i in T.Parallel(remainder):
                    xf = T.Cast("float32", x_ub[base + i])
                    y_ub[base + i] = T.isinf(xf)

    @functools.lru_cache(maxsize=64)
    def _make_isinf_persistent(n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §7).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_isinf_kernel``) only reaches this builder for a shape that does. Serves
        all 5 benchmarked manifest cases.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_isinf_persistent requires n_total ({n_total}) to divide "
            f"num_blocks*tile ({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), "bool")
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _isinf_body(x_ub, y_ub, tile)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_isinf_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not
        evenly divide by ``num_blocks * tile`` (e.g. the 1,048,576-element
        ``test_unary_bool_compile`` compile-smoke fixture, which divides this op's own
        ``tile`` -- 8192/16384 -- exactly but not ``64 * tile``). One block, a main loop
        over full ``tile``-sized chunks plus one smaller tail chunk, mirroring
        ``relu.py``'s/``isnan.py``'s generic single-block fallback shape. Not
        performance-tuned -- no benchmarked manifest case reaches this path (schedule doc
        §7's divisibility table).
        """
        full_tiles, remainder = divmod(n_total, tile)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), "bool")):
            with T.Kernel(1):
                if full_tiles:
                    x_ub = T.alloc_shared((tile,), dtype)
                    y_ub = T.alloc_shared((tile,), "bool")
                    T.annotate_buffer_versions({x_ub: 2, y_ub: 2})
                    for step in T.Pipelined(full_tiles, num_stages=2):
                        begin = step * tile
                        T.copy(X[begin : begin + tile], x_ub)
                        _isinf_body(x_ub, y_ub, tile)
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), "bool")
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _isinf_body(xt_ub, yt_ub, remainder)
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_isinf_kernel(n_total: int, dtype: str):
        """Pick the tiling for ``n_total`` elements of ``dtype``: persistent grid when it
        divides ``num_blocks * tile`` exactly (schedule doc §7), else the generic
        single-block fallback.
        """
        if n_total <= 0:
            raise ValueError(f"n_total must be positive, got {n_total}")
        num_blocks = _calibrated_num_blocks()
        tile = _TILE_FOR_DTYPE[dtype]
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_isinf_persistent(n_total, dtype, num_blocks, tile, _NUM_STAGES)
        return _make_isinf_generic_single_block(n_total, dtype, tile)


class AscendIsinfKernel(Kernel, UnaryPredicateFwdInterface):
    """IsInf on Ascend NPUs: ``y = isinf(x)``, a new contiguous ``torch.bool`` tensor.

    Implements ``UnaryPredicateFwdInterface`` (**not** ``UnaryElementwiseFwdInterface`` --
    see this module's docstring for why that distinction is load-bearing for
    ``IsinfFwdOp``'s own ``interfaces`` override). ``devices`` is NPU only;
    ``supported_archs`` is left ``None`` -- there is no Ascend SoC-version gate here to
    replace the CUDA SM-version one (``Kernel._check_arch`` skips the check entirely when
    ``supported_archs is None``).

    Accepts the op's full 9-dtype manifest domain, not just the 3 floats this kernel
    compiles a TileLang program for (module docstring): ``refusal()`` returns ``None``
    unconditionally, and ``forward()`` branches internally -- the 3 float dtypes
    (``float16``, ``bfloat16``, ``float32``) run the compiled TileLang program below; the
    other 6 (``bool`` plus the 5 int dtypes, none of which can represent +-Inf) are
    answered directly with ``torch.zeros(input.shape, dtype=torch.bool,
    device=input.device)``, bit-identical to the CUDA ``AlwaysFalseFwdKernel.handler`` and
    built with no TileLang compile at all.
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: ElementwiseCall) -> "str | None":
        # Accepts the FULL manifest domain (9 dtypes), not just the 3 floats this kernel
        # actually compiles a program for -- see module docstring: a narrow refusal()
        # here would leave npu+int/bool calls with NO available kernel at all
        # (AlwaysFalseFwdKernel/IsinfFwdKernel are both cuda-only and refuse on device
        # too), breaking test_predicate_non_float_constant on npu.
        return None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendIsinfKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._compiled_fn = (
            _make_isinf_kernel(n_total, _dtype_str(dtype)) if dtype in _FLOAT_DTYPES else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``isinf(input)`` elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements on an NPU device, contiguous in whatever
                shape the caller passed, in ``self.dtype``.

        Returns:
            A new contiguous ``torch.bool`` tensor on the same device, shaped as
            ``input``. For the 6 non-float manifest dtypes (none of which can represent
            +-Inf), this is always all-``False`` -- the constant-output path, no TileLang
            kernel involved (module docstring).
        """
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got {input.device}. "
                "Another target's backend serves other devices."
            )
        if self.dtype not in _FLOAT_DTYPES:
            return torch.zeros(input.shape, dtype=torch.bool, device=input.device)
        flat_input = input.reshape(-1)
        if flat_input.numel() != self.n_total:
            raise ValueError(
                f"{type(self).__name__} was built for n_total={self.n_total}, got "
                f"{flat_input.numel()} elements"
            )
        flat_output = self._compiled_fn(flat_input)
        return flat_output.reshape(input.shape)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("IsinfFwdOp", "ascend", AscendIsinfKernel)
except ImportError:
    pass
