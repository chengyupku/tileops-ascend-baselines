"""Ascend NPU floor kernel: ``y = floor(x)``, via the ``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``FloorFwdOp``; see
``docs/ascend950/dev/FloorFwdOp/round_1_schedule.md`` for the full tiling/schedule design this
file implements. Registers an additional implementation of ``FloorFwdOp``'s
``UnaryElementwiseFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
keys ``"floor"`` (``tileops.kernels.elementwise.math_unary.FloorFwdKernel``) and ``"floor_int"``
(``IntIdentityFwdKernel``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``FloorFwdOp`` instance
constructed after this module is imported (``dev_process.md`` §1).

Math: ``FloorFwdKernel.op_func`` is ``T.floor(T.cast(x, "float32"))`` -- the existing CUDA kernel
already promotes to fp32 unconditionally (its own docstring: "``hfloor`` is not available for
``cutlass::half_t`` in CUDA"). Round 1 mirrors this exactly for all three float dtypes:
``T.cast(T.floor(T.cast(x, "float32")), x.dtype)``, the same ``exp.py``/``sqrt.py`` idiom
(code-style.md: "Promote overflow-prone fp16/bf16 math ... to fp32; cast back to storage dtype
at the boundary") -- though here the promotion is a compile-dispatch workaround, not a precision
fix: ``floor`` is exact at every finite value regardless of storage width.

**Round 1 does not attempt a native 16-bit ``T.floor`` dispatch.**
``tilelang/src/ascend/codegen/intrin_rule_ascend.cc`` registers ``floor``/``ceil``/``round``/
``trunc`` via the identical ``REGISTER_ASCEND_INTRIN`` macro as ``fabs``/``sqrt``/``exp``/``tanh``
-- literally adjacent table rows, with the same purely bit-width-keyed ``AscendMath`` dispatch
(16-bit -> ``"h" + name``, i.e. ``hfloor``; 32-bit -> ``name + "f"``, i.e. ``floorf``; 64-bit ->
bare ``name``) and no per-op special-casing. ``AbsFwdOp``'s worker already drove a real pre-check
at this exact macro table and found two different concrete failure modes for ``T.abs`` at 16-bit:
fp16 compiles down to ``.asc`` C++ and only then fails at the ``bisheng`` backend-compile step
(``error: use of undeclared identifier 'hfabs'``); bf16 fails earlier, at TIR lowering, before any
C++ is generated (``InternalError: Unresolved call ir.Op(... name="tirx.fabs" ...)``, no backend
lowering registered for that storage type at all). A standalone pure-compile pre-check for this op
(module docstring continues in ``round_1_result.md`` §2; no NPU touch, ``tilelang.compile(
target="ascend")`` alone does not initialize the device) confirmed the identical two failure
modes for ``T.floor`` at fp16/bf16 on this toolchain -- the same toolchain/registration gap,
independent of which particular math function is dispatched. fp32's own path (``floorf``) is the
same 32-bit branch ``example_per_token_cast_to_fp8.py`` already proves live for ``fabsf``, so
there is no residual risk at fp32. Round 1's design is therefore: promote unconditionally to fp32
for all three dtypes and never call ``T.floor`` directly on a 16-bit value -- both the behavior
the existing CUDA kernel already has (for an unrelated, CUDA-specific reason) and the safe side of
a toolchain gap this project has already observed on a structurally identical registration table.

Tiling design (schedule doc §4) reuses ``ExpFwdOp``'s round-1 tiling verbatim, since ``floor``'s
per-element body shape (cast in, one intrinsic call, cast out) is the structural match for
``ExpFwdOp``'s body, not ``AbsFwdOp``'s/``NegFwdOp``'s no-promotion bodies:

- ``num_blocks``: ``_calibrated_num_blocks`` (``tilelang.carver.arch.driver.
  get_num_vector_cores(device_id=0)``, floored to a power of two, fallback 64) -- already
  confirmed 64 on this SKU (72 real vector cores) by ``GeluFwdOp``/``SiluFwdOp``/``ReluFwdOp``
  round 2 plus ``ExpFwdOp``/``AbsFwdOp``/``NegFwdOp`` reusing it unchanged; not re-swept here.
- Tile size: ``_UB_BUDGET_BYTES = 216 * 1024`` and ``max_elems = _UB_BUDGET_BYTES //
  (2 * num_stages * itemsize)``, floored to a power of two -- 16384 elements for fp16/bf16, 8192
  for float32, invariant to which ``num_stages`` candidate below is chosen (schedule doc §4's
  arithmetic table).
- ``num_stages``: round 1's one open parameter -- built and measured as both 2 and 3 under the
  official ``measure_tileops_ascend.py`` protocol (not ``msprof_detail``, which flushes the GM
  cache every repeat and has already once produced a wrong ranking for this exact kind of sweep,
  ``ReluFwdOp`` round 2); see ``round_1_result.md`` §2 for the comparison. ``_NUM_STAGES`` below
  is the winner.
- Threads: ``min(tile_elems, 2048)``, the hard Ascend SIMT ceiling (``dev_process.md`` §7.5).
- Small-``n`` routing: a single-block, no-pipelining path below ``_SINGLE_BLOCK_THRESHOLD``,
  keyed off the *narrower* (float32) tile value (8192) so the routing decision does not itself
  depend on dtype -- the same convention ``SqrtFwdOp`` uses for its own per-dtype tile. A
  defensive tail-handling fallback (``_make_floor_generic_single_block``, mirroring ``abs.py``'s
  equivalent) covers any shape dividing neither the tiled path nor the single-block path; no
  manifest/test shape this round exercises (64, 1,048,576, 2**24, 2**28) is expected to reach it.

**Integer dtypes -- a correctness requirement, not a performance one** (schedule doc §3).
``FloorFwdOp.kernel_types`` (``ops/elementwise/math_unary.py``) is ``{"floor": FloorFwdKernel,
"floor_int": IntIdentityFwdKernel}``. ``IntIdentityFwdKernel(TorchFallbackKernel,
UnaryElementwiseFwdInterface, RoundFwdInterface)`` does not override ``Kernel.devices`` (stays
``frozenset({"cuda"})``) and is therefore ``unavailable()`` on an ``npu`` call -- the same pattern
already confirmed for ``AbsIntFwdKernel``/``NegIntFwdKernel``/``SignIntFwdKernel``
(``dev_process.md`` §7.5). If ``AscendFloorKernel.refusal()`` only accepted the 3 benchmarked
float dtypes, an ``npu`` + int call would find zero eligible implementations and ``key_for``
would raise ``ValueError: no implementation serves this call``. ``refusal()`` therefore accepts
the op's full manifest domain (the 3 floats plus ``INT_DTYPES``, imported -- not redefined --
from ``tileops.kernels.elementwise.math_unary``, the same tuple ``IntIdentityFwdKernel.
SUPPORTED_DTYPES`` already uses on the CUDA side), mirroring ``AbsFwdOp``'s ``refusal()`` shape
(an explicit domain check), not ``NegFwdOp``'s unconditional ``return None``. ``forward()``
branches internally: floats run the TileLang program below; ints return
``input.clone().contiguous()`` -- bit-identical to ``IntIdentityFwdKernel.handler``
(``torch.Tensor.clone``) composed with ``TorchFallbackKernel.forward()``'s own ``.contiguous()``
call. Confirmed by a real M3 run (``test_rounding_op_int_identity[FloorFwdOp-*]``, 5 int dtypes),
not assumed -- see ``round_1_result.md`` §3: ``torch.Tensor.clone`` is a generic memcpy, not a
dtype-specific ACLNN compute op, so (unlike ``AscendNegKernel``'s int16/uint8 coverage gap) there
is no ACLNN "support list" to fail against.

Does not inherit ``FloatUnaryKernel``/``UnaryKernel`` (``tileops/kernels/elementwise/_base.py``):
both hardcode CUDA-only behavior (``devices={"cuda"}``, ``supported_archs`` read through
``tileops.utils.get_sm_version``, and ``_require_cuda``'s ``tensor.is_cuda`` check) that would
reject every NPU tensor outright.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, several ops' workers write into this same
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
    # tuple IntIdentityFwdKernel.SUPPORTED_DTYPES already uses on the CUDA side (schedule §3).
    _FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = 64) -> int:
        """Persistent-grid block count for the large, bandwidth-bound tiled path.

        Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
        floors it to the nearest power of two, matching ``ExpFwdOp``'s/``ReluFwdOp``'s
        ``_calibrated_num_blocks`` exactly -- already converged on 64 across multiple
        independent pilots on this SKU (72 real vector cores, ``Ascend950DT_9592``); not
        re-swept here (schedule doc §4).

        Deliberately **not** evaluated at module import time: the query touches the real device,
        so merely importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
        ``lru_cache`` makes it a one-time query per process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    # SIMT UB budget per vector core, in bytes. Reused verbatim from AscendSiluKernel/
    # AscendExpKernel, which discovered it via a real on-device compile failure on float32's
    # dynamic-shared-memory limit -- this op has the identical per-element UB footprint shape
    # (one input tile + one output tile, no extra scratch), so the same budget applies unchanged.
    _UB_BUDGET_BYTES = 216 * 1024

    # Round 1's one open parameter (schedule doc §4): built and measured as both 2 and 3 under
    # the official measure_tileops_ascend.py protocol; see round_1_result.md §2 for the
    # comparison. The resulting tile size (below) is invariant to this choice for both dtype
    # classes this op sees.
    _NUM_STAGES = 3

    # Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to
    # compile.
    _MAX_THREADS = 2048

    def _tile_elems_for_dtype(dtype_str: str) -> int:
        """The largest power-of-two element count keeping the tiled path's UB use in budget.

        2 buffers (x tile, y tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
        ``_UB_BUDGET_BYTES``. Resolves to 16384 for fp16/bf16 (itemsize 2) and 8192 for float32
        (itemsize 4) at both ``num_stages`` candidates this round considers (schedule doc §4).
        """
        itemsize = getattr(torch, dtype_str).itemsize
        max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
        return _floor_pow2(max_elems)

    # Below this many elements, a persistent multi-block grid and T.Pipelined staging would add
    # fixed per-launch overhead with no bandwidth-bound tail to amortize it against (this op has
    # no decode-sized manifest case). Fixed at the *narrower* (float32) tile value, 8192, rather
    # than the per-dtype _tile_elems_for_dtype result, so the routing decision does not itself
    # depend on dtype (the same convention SqrtFwdOp uses for its own per-dtype tile). The one
    # required small shape (test_multi_dtype_instance's n_total=64 sweep) is far below either
    # tile candidate, so this threshold's exact value does not interact with the dtype split.
    _SINGLE_BLOCK_THRESHOLD = 8192

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _floor_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = floor(x_ub[:n])``, computed in fp32, compiler-vectorized over
        ``threads`` lanes.

        Mirrors ``FloorFwdKernel.op_func`` (``kernels/elementwise/math_unary.py``) exactly:
        ``T.floor(T.cast(x, "float32"))``, cast back to the storage dtype at the store boundary.
        Promotion here is a compile-dispatch workaround (``T.floor`` fails to compile directly
        at fp16/bf16 on this toolchain, module docstring), not a precision fix -- ``floor`` is
        exact at every finite value regardless of storage width.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.floor(T.cast(x_ub[i], "float32")), y_ub.dtype)

    @functools.lru_cache(maxsize=64)
    def _make_floor_single_block(n_total: int, dtype: str):
        """One block, one tile, no pipelining: correctness-first path for small ``n_total``
        (below ``_SINGLE_BLOCK_THRESHOLD``) -- this op's manifest has no decode-sized benchmark
        case, but ``test_multi_dtype_instance``'s cross-op smoke sweep still drives every float
        dtype at ``n_total=64``, which needs a working small-``n`` path.
        """

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(1):
                x_ub = T.alloc_shared((n_total,), dtype)
                y_ub = T.alloc_shared((n_total,), dtype)
                T.copy(X, x_ub)
                _floor_body(x_ub, y_ub, n_total, threads=min(_MAX_THREADS, n_total))
                T.copy(y_ub, Y)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_floor_persistent(
        n_total: int, dtype: str, num_blocks: int, tile: int, num_stages: int
    ):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §4).

        Requires ``n_total`` to divide ``num_blocks * tile`` exactly; the caller
        (``_make_floor_kernel``) only reaches this builder for a shape that does, and routes
        every other shape to ``_make_floor_generic_single_block`` instead. All 5 of this op's
        benchmarked cases, plus the correctness fixture's ``n_total=1_048_576``, hit this path.
        """
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_floor_persistent requires n_total ({n_total}) to divide "
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
                    _floor_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    @functools.lru_cache(maxsize=64)
    def _make_floor_generic_single_block(n_total: int, dtype: str, tile: int):
        """Correctness-first fallback for an ``n_total`` the persistent tiling does not evenly
        divide (schedule doc §4: "write it anyway for defensiveness/future shapes"). One block;
        a main loop over full tile-sized chunks, plus one smaller, statically-sized tail chunk.
        Not performance-tuned: none of this op's manifest cases or required test fixtures reach
        this function this round (every shape this round exercises -- 64, 1_048_576, 2**24,
        2**28 -- either fits the single-block path or divides ``num_blocks * tile`` exactly),
        mirroring ``abs.py``'s ``_make_abs_generic_single_block``.
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
                        _floor_body(x_ub, y_ub, tile, threads=min(_MAX_THREADS, tile))
                        T.copy(y_ub, Y[begin : begin + tile])
                if remainder:
                    xt_ub = T.alloc_shared((remainder,), dtype)
                    yt_ub = T.alloc_shared((remainder,), dtype)
                    begin = full_tiles * tile
                    T.copy(X[begin : begin + remainder], xt_ub)
                    _floor_body(xt_ub, yt_ub, remainder, threads=min(128, remainder))
                    T.copy(yt_ub, Y[begin : begin + remainder])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _make_floor_kernel(n_total: int, dtype_str: str):
        """Pick the tiling for ``n_total`` elements: single-block, persistent, or fallback.

        ``_SINGLE_BLOCK_THRESHOLD`` is fixed (not per-dtype), so the routing decision itself
        does not depend on dtype; the per-dtype tile only matters once the persistent path is
        reached (schedule doc §4).
        """
        if n_total <= _SINGLE_BLOCK_THRESHOLD:
            return _make_floor_single_block(n_total, dtype_str)
        tile = _tile_elems_for_dtype(dtype_str)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile
        if n_total % block_span == 0:
            return _make_floor_persistent(n_total, dtype_str, num_blocks, tile, _NUM_STAGES)
        return _make_floor_generic_single_block(n_total, dtype_str, tile)

    class AscendFloorKernel(Kernel, UnaryElementwiseFwdInterface):
        """Floor on Ascend NPUs: ``y = floor(x)``, computed in fp32 for all float dtypes.

        See ``docs/ascend950/dev/FloorFwdOp/round_1_schedule.md`` for the tiling design.
        Memory-bound, non-transcendental, exact at every finite value regardless of storage
        width -- fp32 promotion here is purely a compile-dispatch workaround for a confirmed
        16-bit toolchain gap (module docstring), not a precision fix. A fresh class per
        ``dev_process.md`` §1 -- it does not subclass the CUDA-locked ``UnaryKernel``/
        ``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the CUDA dialect
        facade).

        Accepts the op's *full* manifest domain -- the 3 benchmarked float dtypes plus the 5
        integer dtypes the manifest also admits -- not just the floats (schedule doc §3):
        ``forward()`` runs the TileLang program above for float input, and for integer input
        returns ``input.clone().contiguous()``, bit-identical to ``IntIdentityFwdKernel.
        handler`` (``torch.Tensor.clone``), confirmed by a real M3 run to need no ACLNN
        promote/truncate workaround (unlike ``AscendNegKernel``'s int16/uint8 gap) -- a generic
        memcpy has no narrow-coverage ACLNN arithmetic op to fail against.
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
            if self._is_int:
                self._compiled_fn = None
            else:
                self._compiled_fn = _make_floor_kernel(n_total, _dtype_str(dtype))

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                # Bit-identical to IntIdentityFwdKernel.handler (math_unary.py:291): an integer
                # is already its own floor. UnaryOp.forward() already made `input` contiguous
                # before this call (ops/elementwise/_base.py); the extra .contiguous() is cheap
                # defensive parity with TorchFallbackKernel.forward()'s own explicit call.
                return input.clone().contiguous()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("FloorFwdOp", "ascend", AscendFloorKernel)

except ImportError:
    pass
