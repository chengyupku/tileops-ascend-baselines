"""Ascend NPU kernel for ``CeilFwdOp``: elementwise ``y = ceil(x)``.

Registers an additional implementation of ``CeilFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA keys ``"ceil"``
(``tileops.kernels.elementwise.math_unary.CeilFwdKernel``) and ``"ceil_int"``
(``IntIdentityFwdKernel``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``CeilFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/CeilFwdOp/round_1_schedule.md``. Math mirrors ``CeilFwdKernel.
op_func`` (``kernels/elementwise/math_unary.py``) exactly: ``T.ceil(T.cast(x, "float32"))`` --
cast to fp32, call the hardware ``ceil`` intrinsic, cast back to the storage dtype at the shared
store boundary. This is not just CUDA-habit copying: the schedule traced three stack layers
(``tilelang/language/tir/ir.py`` -> ``tilelang.ascend.language`` ->
``src/ascend/codegen/intrin_rule_ascend.cc``) and confirmed ``T.ceil`` is dispatched through the
same ``AscendMath`` bit-width switch (``DispatchPureExtern<AscendMath>``) already used by
``sqrt``/``exp``/``tanh``/``rsqrt`` -- a dispatcher that does not distinguish float16 from
bfloat16 at 16 bits (both resolve to the literal string ``"hceil"``) and has already hard-failed
bf16 on sibling intrinsics (``"Unresolved call ir.Op(...)"``, dev_process.md §7.5) because the
device-side template headers only overload ``h<name>`` for ``half_t``, not ``bfloat16_t``.
Casting to fp32 first makes ``t.bits()==32`` unconditionally, so ``AscendMath`` always emits
``ceilf`` and the 16-bit branch (and its bf16 gap) is never reached -- for any of the three float
dtypes. A native 16-bit direct-dispatch path is deliberately **not** attempted this round: the
closest structural analog, ``TanhFwdOp`` round 2, built and measured a native fp16 path (skipping
the fp32 promote, since fp16 -- unlike bf16 -- compiles) and found it 3.4-8% *slower* than the
promote-to-fp32 path on both wall-clock and hardware AIV-cycle measurements. ``ceil`` is an even
cheaper single-instruction op than ``tanh``, so there is no reason to expect that economics to
flip here.

Tiling design (schedule doc §2) reuses the fully-converged persistent-grid constants verbatim
from this batch's precedent kernels (``ExpFwdOp``, ``NegFwdOp``, ``AbsFwdOp``) rather than
re-deriving them: ``num_blocks=64`` (``tilelang.carver.arch.driver.get_num_vector_cores``,
floored to a power of two, fallback 64 -- this SKU's real count, ``Ascend950DT_9592``, is 72),
216 KiB UB budget giving ``tile_elems=16384`` for fp16/bf16 and ``8192`` for float32, threads
``min(tile_elems, 2048)``. Per the schedule's divisibility table (§2.1), every shape this op's
test/benchmark surface actually exercises (5 manifest cases, the ``MathFixture`` M3 shapes, the
compile-smoke shape) divides ``num_blocks * tile_elems`` exactly -- unlike ``ExpFwdOp``, round 1
needs **zero** fallback/tail/single-block path here, matching ``NegFwdOp``'s precedent instead.
``_build_tiled`` therefore asserts divisibility rather than branching to a second builder.

``num_stages`` is round 1's one open parameter (schedule doc §2.4): built and measured as both 2
and 3, under the official ``measure_tileops_ascend.py`` protocol (not ``msprof_detail``, which
flushes the GM cache every repeat and has already produced one wrong ranking on this exact kind
of sweep, ``ReluFwdOp`` round 2, dev_process.md §7.5). See ``round_1_result.md`` §4 for the
measured comparison; ``_NUM_STAGES`` below is the winner actually shipped.

**Integer dtypes -- a correctness requirement, not a performance one** (schedule doc §3): the
manifest's ``forall`` also admits 5 integer dtypes (uint8/int8/int16/int32/int64), served on CUDA
by ``IntIdentityFwdKernel(TorchFallbackKernel)`` (``handler = torch.Tensor.clone`` -- ceiling an
already-integral value is a no-op by definition). That class does not override ``Kernel.devices``
(stays ``frozenset({"cuda"})``), so it is ``unavailable()`` on an ``npu`` call. If
``AscendCeilKernel.refusal()`` only accepted float dtypes, an ``npu`` + int call would find zero
eligible implementations and ``key_for`` would raise ``ValueError: no implementation serves this
call`` -- breaking ``tests/ops/test_unary_math.py::test_rounding_op_int_identity[CeilFwdOp-*]``
(shared with ``FloorFwdOp``/``RoundFwdOp``/``TruncFwdOp``), for reasons having nothing to do with
this op's own 5 benchmark cases. ``refusal()`` therefore accepts the full manifest domain (same
pattern ``AscendNegKernel.refusal``/``ReciprocalFwdKernel.refusal`` already use), and ``forward()``
branches internally: the 3 float dtypes run the TileLang program below; the 5 int dtypes skip
TileLang entirely and return ``input.clone()`` -- bit-for-bit what ``IntIdentityFwdKernel.handler``
already computes on CUDA. Unlike ``NegFwdOp`` (whose NPU ``aclnnNeg`` did not implement every int
width in its ACL support list and needed an int32 promote/truncate workaround for int16/uint8),
``torch.Tensor.clone`` is a storage copy, not an arithmetic ACL kernel with a declared dtype
support list -- there is no a-priori reason to expect a per-width gap, and
``test_rounding_op_int_identity`` confirms this live (round_1_result.md §3) rather than assuming
it from the schedule's reasoning.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call are
all wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md`` §1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``docs/ascend950/dev_process.md`` §1, several ops' workers write into this same
``kernels/ascend``/``kernels/ascend/elementwise`` directory concurrently, each registering from
its own file, and a shared ``__init__.py`` would be a file they all contend for. ``kernels/ascend``
and ``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420) namespace packages,
matching every other Ascend elementwise kernel already in this directory.
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

    # The manifest's non-float dtypes for this op (schedule doc §3): CUDA serves these through
    # IntIdentityFwdKernel's torch.Tensor.clone fallback (ceiling an already-integral value is a
    # no-op by definition), which this class's forward() mirrors rather than compiling a TileLang
    # program for them. No bool here -- the manifest does not admit it for CeilFwdOp.
    _INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

    # SIMT UB budget per vector core, in bytes. Reused verbatim from AscendSiluKernel/
    # AscendExpKernel, which discovered it via a real on-device compile failure on float32's
    # dynamic-shared-memory limit -- this op has the identical per-element UB footprint shape
    # (one input tile + one output tile, no extra scratch), so the same budget applies unchanged.
    _UB_BUDGET_BYTES = 216 * 1024
    # The one open round-1 parameter (schedule doc §2.4): built and measured as both 2 and 3
    # under the official measure_tileops_ascend.py protocol; see round_1_result.md §4 for the
    # comparison. This is the value actually shipped.
    _NUM_STAGES = 3
    # Conservative fallback vector-core count if get_num_vector_cores() raises or returns a
    # falsy value. 64 is this SKU's own measured-optimal value (ReluFwdOp/GeluFwdOp/SiluFwdOp
    # round 2, reconfirmed by every batch-3 unary op since), not an arbitrary guess, so a failed
    # query still degrades to the right answer for this device.
    _FALLBACK_VECTOR_CORES = 64
    # Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to
    # compile.
    _MAX_THREADS = 2048

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    def _floor_pow2(n: int) -> int:
        """Largest power of two ``<= n`` (``n >= 1``)."""
        p = 1
        while p * 2 <= n:
            p *= 2
        return p

    @functools.lru_cache(maxsize=1)
    def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
        """Persistent-grid block count for the tiled, bandwidth-bound path.

        Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
        floors it to the nearest power of two, matching every batch-3 sibling kernel's
        ``_calibrated_num_blocks`` exactly. Deliberately **not** evaluated at module import time:
        the query touches the real device, so merely importing this module must stay a
        no-NPU-touch no-op (``dev_process.md`` §4). ``lru_cache`` makes it a one-time query per
        process either way.
        """
        try:
            queried = get_num_vector_cores(device_id=0)
        except Exception:
            queried = None
        if not queried:
            return fallback
        return _floor_pow2(int(queried))

    def _tile_elems_for_dtype(dtype_str: str) -> int:
        """The largest power-of-two element count keeping the tiled path's UB use in budget.

        2 buffers (x tile, y tile) x ``_NUM_STAGES`` x ``itemsize`` x ``tile_elems`` must fit
        ``_UB_BUDGET_BYTES``. Resolves to 16384 for fp16/bf16 (itemsize 2) and 8192 for float32
        (itemsize 4) at both ``num_stages`` candidates this round considers (schedule doc §2.3's
        table shows the result is invariant to that choice here).
        """
        itemsize = getattr(torch, dtype_str).itemsize
        max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
        return _floor_pow2(max_elems)

    @T.macro
    def _ceil_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = ceil(x_ub[:n])``, computed in fp32, compiler-vectorized over ``threads``.

        Mirrors ``CeilFwdKernel.op_func``'s CUDA body exactly: ``T.ceil(T.cast(x, "float32"))``,
        cast back to the storage dtype at the store (code-style.md's fp32-promotion rule; see the
        module docstring for why this also sidesteps a known Ascend bf16 dispatch hazard for
        free). A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's ``for i in
        T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = T.cast(T.ceil(T.cast(x_ub[i], "float32")), y_ub.dtype)

    @functools.lru_cache(maxsize=16)
    def _build_tiled(
        n_total: int, dtype_str: str, num_blocks: int, tile_elems: int, num_stages: int
    ):
        """Persistent-block, ``T.Pipelined`` main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly. Per schedule doc §2.1's
        divisibility table, every shape this op's test/benchmark surface is ever asked to run (5
        manifest cases, the ``MathFixture`` M3 shapes, the compile-smoke shape) satisfies this --
        unlike ``ExpFwdOp``, this op needs no single-block/tail fallback, matching ``NegFwdOp``'s
        precedent instead. The assertion below guards against silently computing a wrong answer
        for a shape outside that set, rather than implementing a real tail this round.
        """
        block_span = num_blocks * tile_elems
        assert n_total % block_span == 0, (
            f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; round 1 ships no tail/decode path (round_1_schedule.md §2.1)"
        )
        iterations = n_total // block_span
        threads = min(tile_elems, _MAX_THREADS)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _ceil_body(x_ub, y_ub, tile_elems, threads=threads)
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_ceil_kernel(n_total: int, dtype_str: str):
        """Build the single persistent-tiled kernel for ``n_total`` elements.

        Per schedule doc §2.1, every float shape this kernel is ever asked to run divides
        ``num_blocks * tile_elems`` exactly, so round 1 ships exactly one kernel-building path
        (no decode/tail branch, no single-block fallback). A defensive ``NotImplementedError``
        covers a hypothetical future shape outside that set.
        """
        tile_elems = _tile_elems_for_dtype(dtype_str)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile_elems
        if n_total % block_span == 0:
            return _build_tiled(n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES)
        raise NotImplementedError(
            f"AscendCeilKernel: n_total={n_total} dtype={dtype_str} does not divide "
            f"num_blocks*tile_elems={block_span}; round 1 ships no tail/decode path "
            "(round_1_schedule.md §2.1) and no manifest/test shape should reach this"
        )

    class AscendCeilKernel(Kernel, UnaryElementwiseFwdInterface):
        """Ascend NPU implementation of ``CeilFwdOp``: ``y = ceil(x)``.

        Memory-bound (one load, one fp32-promoted ``ceil``, one store per element); computed in
        fp32 regardless of storage dtype, mirroring ``CeilFwdKernel.op_func`` exactly -- see the
        module docstring for why this also avoids a known Ascend bf16 ``AscendMath`` dispatch
        hazard, not just a CUDA-habit carryover. A fresh class per
        ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language`` (the
        CUDA dialect facade).

        Accepts the op's *full* manifest domain -- the 3 benchmarked float dtypes plus the 5
        integer dtypes the manifest also admits -- not just the floats (schedule doc §3):
        ``forward()`` runs the TileLang program below for float input, and for integer input
        reproduces ``IntIdentityFwdKernel.handler``'s result bit-for-bit via ``input.clone()``
        (ceiling an already-integral value is a no-op for every width), so an ``npu`` + int call
        still has an eligible implementation once this module is imported.

        Tolerance (inherited from the shared elementwise test tolerance table, not re-derived
        here): fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 (``CeilFwdOp`` is not in any of
        ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the
        default ``Exact()``). ``ceil`` has no accumulation and no transcendental approximation
        error -- every fp16/bf16 input is exactly representable in fp32, so the fp32 round-trip
        is exact in practice for this op's realistic (``_randn``) input domain.
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: ElementwiseCall) -> "str | None":
            # Unconditionally None (same pattern AscendNegKernel.refusal/
            # ReciprocalFwdKernel.refusal use): this class serves every dtype the manifest
            # admits for this op, not just the 3 benchmarked floats. See the class docstring and
            # schedule doc §3 for why a float-only refusal here would regress the shared
            # test_rounding_op_int_identity test.
            return None

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._is_int = dtype in _INT_DTYPES
            self._compiled_fn = (
                None if self._is_int else _build_ceil_kernel(n_total, _dtype_str(dtype))
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                # Bit-for-bit what IntIdentityFwdKernel.handler already computes on CUDA:
                # ceiling an already-integral value is a no-op by definition, for every width
                # (schedule doc §3). UnaryOp.forward() already made `input` contiguous before
                # this call; .clone() preserves that contiguity.
                return input.clone()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("CeilFwdOp", "ascend", AscendCeilKernel)

except ImportError:
    pass
