"""Ascend NPU kernel for ``NegFwdOp``: elementwise ``y = -x``.

Registers an additional implementation of ``NegFwdOp``'s ``UnaryElementwiseFwdInterface``
under the ``"ascend"`` dispatch key, beside the in-tree CUDA keys ``"neg"``
(``tileops.kernels.elementwise.math_unary.NegFwdKernel``) and ``"neg_int"``
(``NegIntFwdKernel``). No existing file is edited to do this:
``tileops.backend.register_kernel_type`` joins this class to every ``NegFwdOp`` instance
constructed after this module is imported (``docs/ascend950/dev_process.md`` §1).

Design: ``docs/ascend950/dev/NegFwdOp/round_1_schedule.md``. The plainest body in the unary-
math family -- one negate, no accumulation or overflow risk at any finite input -- so the
tiling is not re-derived this round: it reuses, unmodified, the persistent-grid config
(``num_blocks=64``, ``tile=8192``, ``num_stages=2``, ``threads=min(tile, 2048)=2048``) that
``ReluFwdOp`` round 2's official (no-cache-flush) M4 sweep already found optimal for this exact
SKU's bandwidth-bound unary shape, independently confirmed by ``BitwiseNotFwdOp`` round 1. All
5 benchmarked cases are pure powers of two >= 2**24, and ``num_blocks * tile = 2**19`` divides
every one of them with zero remainder, so round 1 ships exactly one kernel-building path, with
no decode/tail branch (schedule doc §3/§4).

Negation is exact (a sign-bit flip, no rounding, no accumulation) at every finite fp16/bf16/
fp32 value and propagates NaN/Inf the same way regardless of compute width, so the body
computes directly in the storage dtype -- no fp32 promotion, mirroring ``NegFwdKernel.op_func``
(``return -x``) exactly (schedule doc §3).

**Integer dtypes -- a correctness requirement, not a performance one** (schedule doc §5): the
manifest's ``forall`` also admits 5 integer dtypes (int8/int16/int32/int64/uint8), served on
CUDA by ``NegIntFwdKernel(TorchFallbackKernel)``, which does not override ``Kernel.devices``
(stays ``frozenset({"cuda"})``) and is therefore ``unavailable()`` on an ``npu`` call. If
``AscendNegKernel.refusal()`` only accepted float dtypes, an ``npu`` + int call would find zero
eligible implementations (``NegIntFwdKernel`` refuses the device, this class would refuse the
dtype) and ``key_for`` would raise ``ValueError: no implementation serves this call`` --
breaking two tests shared with other concurrently-planned ops in this batch
(``tests/ops/test_unary_math.py::test_unary_int_torch_fallback``,
``tests/ops/test_elementwise_compile.py::test_compiled_non_contiguous_integer_fallback``), for
reasons having nothing to do with this op's own 5 benchmark cases. ``refusal()`` therefore
accepts the full manifest domain (floats + the 5 ints, same pattern
``ReciprocalFwdKernel.refusal`` already uses on the CUDA side), and ``forward()`` branches
internally: the 3 float dtypes run the TileLang program below; the 5 int dtypes skip TileLang
entirely and are answered via ``torch.neg``, matching ``NegIntFwdKernel.handler``'s result
bit-for-bit.

**Round-1 M3 correction, found by running the actual test, not assumed from the schedule**:
the schedule's plan to call ``torch.neg(input)`` directly for every int dtype is only half
right on this backend. ``torch.neg`` is mathematically well-defined on every int width the
manifest admits (including uint8's modular wraparound), which is what CUDA's
``NegIntFwdKernel`` relies on -- but NPU's own operator (``aclnnNeg``) does not implement that
full width set: a direct ``torch.neg`` call on an NPU int16 or uint8 tensor raises at the ACL
layer (``AclNN_Parameter_Error(EZ1001): Self dtype DT_INT16/DT_UINT8 should be in dtype support
list [DT_INT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
DT_COMPLEX128, DT_BFLOAT16]``), confirmed live by
``test_unary_math.py::test_unary_int_torch_fallback``. int8/int32/int64 are in that support
list and go through ``torch.neg`` unmodified; int16 and uint8 are promoted to int32, negated
there (never overflows: int32 covers every int16/uint8 value with room to spare), and cast back
down. This is bit-exact with a native int16/uint8 negate (and with ``NegIntFwdKernel.handler``
on CUDA): two's-complement truncation on the int32->int16/uint8 downcast reproduces the same
wraparound a native-width negate would have produced (verified both by construction -- negation
and truncation commute under two's-complement arithmetic -- and empirically, 100,000 random
values per dtype against CPU ``torch.neg``, before touching the NPU). bool, which ``torch.neg``
rejects outright, is correctly absent from this op's manifest domain, so that restriction never
applies.

Importing this module when ``tilelang.ascend`` is unavailable (no Ascend toolchain, a CUDA-only
process) is a no-op: the Ascend-specific imports, the kernel class, and the registration call
are all wrapped in one ``try/except ImportError`` block, per ``docs/ascend950/dev_process.md``
§1.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``docs/ascend950/dev_process.md`` §1, several ops' workers write into this same
new ``kernels/ascend``/``kernels/ascend/elementwise`` directory concurrently, each registering
from its own file, and a shared ``__init__.py`` would be a file they all contend for.
``kernels/ascend`` and ``kernels/ascend/elementwise`` are therefore left as implicit (PEP 420)
namespace packages, matching every other Ascend elementwise kernel already in this directory.
"""

import functools

import torch

try:
    import tilelang
    import tilelang.ascend.language as T

    import tileops.backend
    from tileops.kernels.elementwise.call_spec import (
        ElementwiseCall,
        UnaryElementwiseFwdInterface,
    )
    from tileops.kernels.kernel_base import Entry, Kernel

    # The manifest's non-float dtypes for this op (schedule doc §5, §1): CUDA serves these
    # through NegIntFwdKernel's torch.neg fallback, which this class's forward() mirrors
    # rather than compiling a TileLang program for them. No bool here -- the manifest does
    # not admit it for NegFwdOp, and torch.neg rejects it.
    _INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

    # The subset of _INT_DTYPES whose width this NPU build's torch.neg (aclnnNeg) actually
    # implements -- found by running test_unary_math.py::test_unary_int_torch_fallback
    # (round-1 M3), not assumed: aclnnNeg's error-reported support list is
    # [DT_INT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_COMPLEX64,
    # DT_COMPLEX128, DT_BFLOAT16], so int16 and uint8 are missing and need the int32
    # promote/truncate path in forward() instead of a direct call (see the module
    # docstring's "Round-1 M3 correction" paragraph).
    _NATIVE_NPU_NEG_INT_DTYPES = (torch.int8, torch.int32, torch.int64)

    def _dtype_str(dtype: torch.dtype) -> str:
        return str(dtype).split(".")[-1]

    @T.macro
    def _neg_body(x_ub, y_ub, n: int, threads: int) -> None:
        """``y_ub[:n] = -x_ub[:n]``, compiler-vectorized over ``threads`` lanes.

        Mirrors ``NegFwdKernel.op_func``'s CUDA body (``return -x``) exactly: no cast, no
        fp32 promotion -- negation is a sign-bit flip, exact at every finite fp16/bf16/fp32
        value, with no accumulation or overflow risk to guard against (schedule doc §3).

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``"): an undecorated helper's
        ``for i in T.Parallel(n):`` raises ``TypeError: 'ForFrame' object is not iterable``
        under TileLang's eager builder (``dev_process.md`` §7.5).
        """
        with T.SimtVF(threads=threads):
            for i in T.Parallel(n):
                y_ub[i] = -x_ub[i]

    @functools.lru_cache(maxsize=16)
    def _make_neg_kernel(n_total: int, dtype: str, tile: int):
        """Persistent multi-block grid with ``T.Pipelined`` staging (schedule doc §3).

        ``num_blocks=64``/``num_stages=2`` stay locals here (code-style.md: a value only
        one function reads is a local) -- both already independently confirmed best-of-
        swept for this persistent-grid family on two sibling ops (``ReluFwdOp`` round 2,
        ``BitwiseNotFwdOp``) and not re-litigated by round 2's tile sweep. ``tile`` is a
        parameter, not a hardcoded local, since round 2 (``round_2_schedule.md`` §2) sweeps
        it across ``{8192, 4096, 2048}`` (varying ``T.Pipelined`` iteration count at fixed
        ``num_blocks``/``num_stages``) before ``AscendNegKernel.__init__`` settles on one
        value to ship (``round_2_result.md``).

        Raises:
            AssertionError: ``n_total`` does not divide ``num_blocks * tile`` evenly. None
                of this round's 5 benchmarked cases hits this, at any of the swept ``tile``
                values; it guards against silently computing a wrong answer for a shape
                outside that set rather than handling a tail this round (schedule doc §4).
        """
        num_blocks, num_stages = 64, 2
        block_span = num_blocks * tile
        assert n_total % block_span == 0, (
            f"_make_neg_kernel requires n_total ({n_total}) to divide num_blocks*tile "
            f"({block_span}) exactly; round 1 ships no tail/decode path "
            "(round_1_schedule.md §3/§4)"
        )
        iterations = n_total // block_span
        threads = min(tile, 2048)

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype), Y: T.Tensor((n_total,), dtype)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile,), dtype)
                y_ub = T.alloc_shared((tile,), dtype)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile
                    T.copy(X[begin : begin + tile], x_ub)
                    _neg_body(x_ub, y_ub, tile, threads=threads)
                    T.copy(y_ub, Y[begin : begin + tile])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    class AscendNegKernel(Kernel, UnaryElementwiseFwdInterface):
        """Ascend NPU implementation of ``NegFwdOp``: ``y = -x``.

        Memory-bound (one load, one sign-bit flip, one store per element); see
        ``docs/ascend950/dev/NegFwdOp/round_1_schedule.md`` §3 for why the body stays in
        the storage dtype (no fp32 promotion) and for the tiling rationale. A fresh class
        per ``docs/ascend950/dev_process.md`` §1 -- it does not subclass the CUDA-locked
        ``UnaryKernel``/``FloatUnaryKernel`` family, which imports ``tilelang.language``
        (the CUDA dialect facade).

        Accepts the op's *full* manifest domain -- the 3 benchmarked float dtypes plus the
        5 integer dtypes the manifest also admits -- not just the floats (schedule doc §5):
        ``forward()`` runs the TileLang program below for float input, and for integer input
        reproduces ``NegIntFwdKernel.handler``'s result bit-for-bit via ``torch.neg`` -- direct
        for int8/int32/int64 (this NPU build's ``aclnnNeg`` implements those widths), via an
        int32 promote/negate/truncate round-trip for int16/uint8 (``aclnnNeg`` does not
        implement those two directly on this build; see the module docstring's "Round-1 M3
        correction") -- so an ``npu`` + int call still has an eligible implementation once this
        module is imported.
        """

        devices = frozenset({"npu"})

        @classmethod
        def refusal(cls, call: ElementwiseCall) -> "str | None":
            # Unconditionally None (same pattern ReciprocalFwdKernel.refusal uses on the
            # CUDA side): this class serves every dtype the manifest admits for this op,
            # not just the 3 benchmarked floats. See the class docstring and schedule doc
            # §5 for why a float-only refusal here would regress shared int-dtype tests.
            return None

        @classmethod
        def entry_for(cls, call: ElementwiseCall) -> Entry:
            return call, lambda: cls(call.n_total, call.dtype)

        def __init__(self, n_total: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.n_total = n_total
            self.dtype = dtype
            self._is_int = dtype in _INT_DTYPES
            # tile=8192: round 1's config. Round 2 (round_2_schedule.md §2) swept
            # {4096, 2048} under the official M4 protocol as the untried "more
            # T.Pipelined iterations" lever; neither candidate was a net win (4/5 cases
            # regressed sharply -- 13-77% slower -- and the lone case that improved,
            # 16M-float32, still missed the 80% target even at its best tile). See
            # round_2_result.md for the full sweep data; this value is unchanged from
            # round 1.
            tile = 8192
            self._compiled_fn = (
                None
                if self._is_int
                else _make_neg_kernel(n_total, _dtype_str(dtype), tile)
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            if self._is_int:
                if input.dtype in _NATIVE_NPU_NEG_INT_DTYPES:
                    # torch.neg directly -- bit-for-bit what NegIntFwdKernel.handler
                    # already computes for this dtype (schedule doc §5), and this NPU
                    # build's aclnnNeg does implement this width (confirmed live by
                    # test_unary_math.py::test_unary_int_torch_fallback).
                    result = torch.neg(input)
                else:
                    # int16/uint8: aclnnNeg does not implement these widths on this NPU
                    # build (see the module docstring's "Round-1 M3 correction" -- found
                    # by running the test, not assumed). Promote to int32 (always wide
                    # enough, never overflows), negate there, and truncate back: this is
                    # bit-exact with a native-width negate because two's-complement
                    # negation and truncation commute (verified by construction and by
                    # a 100,000-sample CPU cross-check against torch.neg per dtype).
                    result = torch.neg(input.to(torch.int32)).to(input.dtype)
                # UnaryOp.forward() already made `input` contiguous before this call; the
                # extra .contiguous() is cheap defensive parity with
                # TorchFallbackKernel.forward()'s own explicit call, not a correctness
                # requirement.
                return result.contiguous()
            flat = input.reshape(-1)
            result = self._compiled_fn(flat)
            return result.reshape(input.shape)

    tileops.backend.register_kernel_type("NegFwdOp", "ascend", AscendNegKernel)

except ImportError:
    pass
