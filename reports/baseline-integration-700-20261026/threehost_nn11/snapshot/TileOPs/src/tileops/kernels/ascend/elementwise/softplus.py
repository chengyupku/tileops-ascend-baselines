"""Ascend NPU softplus kernel: threshold-guarded ``y = log(1 + exp(beta*x)) / beta`` else ``x``,
via the ``tilelang.ascend`` dialect.

Round 1 of ``docs/ascend950/dev_process.md``'s Ascend development loop for ``SoftplusFwdOp``; see
``docs/ascend950/dev/SoftplusFwdOp/round_1_schedule.md`` for the full tiling/schedule design this
file implements. Registers an additional implementation of ``SoftplusFwdOp``'s
``SoftplusFwdInterface`` under the ``"ascend"`` dispatch key, beside the in-tree CUDA
``SoftplusFwdKernel`` (``tileops.kernels.elementwise.activations.SoftplusFwdKernel``). No existing
file is edited to do this: ``tileops.backend.register_kernel_type`` joins this class to every
``SoftplusFwdOp`` instance constructed after this module is imported (``dev_process.md`` §1).

Math (schedule doc §2): ``beta``/``threshold`` are op-construction compile-time constants (plain
Python floats closed over at trace time, confirmed via ``SoftplusCall``'s dataclass fields and the
CUDA ``_make_softplus_kernel``'s ``functools.lru_cache``-on-``(N, dtype, beta, threshold, ...)``
signature -- never a runtime tensor/scalar load). The body mirrors ``_make_softplus_kernel``
(``kernels/elementwise/activations.py``) exactly: compute ``scaled = beta * x``,
``exp(scaled)`` and ``log(1 + exp(scaled)) / beta`` **unconditionally for every element**, then
``T.if_then_else(scaled > threshold, x, composite)``. This is safe because IEEE-754 ``exp``
saturates to ``+inf`` on overflow rather than trapping or producing NaN (independently confirmed
on this exact hardware by ``ExpFwdOp``'s round-1 worker: bit-exact ``exp(200.0) == +inf``,
``isnan`` false), so the discarded branch for an extreme input is ``log(1+inf)=inf``, never NaN,
and the select still resolves to the correct ``x`` branch cleanly. ``T.if_then_else`` is already
confirmed to compile and pass M3/M4 on Ascend (``AscendReluKernel._relu_body``).

**Deliberately does not use ``log_for_output_precision``'s ``T.__log`` fp16/bf16 branch**
(``kernels/elementwise/_dtype.py``): ``T.__log`` is a CUDA-only fast-math dunder intrinsic
(exported only from ``tilelang.cuda.language``; ``tilelang/testing/python/language/
test_tilelang_language_dialect.py`` asserts ``"__log" not in T_comm.__all__`` and
``"__log" in cuda_language.__all__``) that does not exist in the Ascend dialect at all --
``tilelang.ascend.language`` has no ``__log`` attribute. This file uses plain ``T.log`` (fp32-
promoted) for **every** dtype instead, which is both the only Ascend-valid option and the
cheaper one: ``Log1pFwdOp``'s round-1 finding (``docs/ascend950/dev/Log1pFwdOp/round_1_result.md``
§5, ``dev_process.md`` §7.5) is that this toolchain's device ``log1pf`` is a ~25-instruction
software polynomial (bit-level range reduction, an 8-term FMA chain, explicit special-value
handling), 2.3-2.8x more expensive per call than the ``logf(1+x)`` composite -- and this op's own
natural formula already *is* that cheaper composite form, so there is no scenario in this design
that would ever reach for genuine ``T.log1p``.

Tiling design (schedule doc §3) reuses the same UB-budget formula and vector-core count that
``AscendExpKernel``/``AscendLog1pKernel`` converged on (both single-input, single-output,
two-buffer elementwise kernels -- the intermediate ``scaled``/``e``/``sp`` values here are
per-thread registers, not UB allocations, so they do not change the UB budget arithmetic even
though this op's body has more arithmetic than either sibling):

- ``num_blocks``: live-queried vector-core count (``get_num_vector_cores``), floored to a power
  of two, fallback 64 -- five-plus independent pilots on this SKU (``Ascend950DT_9592``, 72 real
  vector cores) already converged on 64 as both the live-queried floor and the measured optimum.
- Tile size: the standard UB-budget formula (``_UB_BUDGET_BYTES = 216 * 1024``, 2 buffers,
  itemsize-scaled) gives 16384 elements for float16/bfloat16, 8192 for float32, confirmed by hand
  invariant to the ``num_stages`` choice below (schedule doc §3.2's table).
- Threads: ``min(tile_elems, 2048)`` = 2048 for both tile sizes (both exceed the SIMT ceiling).

``num_stages`` is round 1's one open parameter (schedule doc §3.3): built and measured as both 2
and 3 under the official ``measure_tileops_ascend.py`` protocol (not ``msprof_detail``, which
flushes the GM cache every repeat and has previously produced a wrong ranking for this exact kind
of sweep -- ``dev_process.md`` §7.5); see ``round_1_result.md`` §4 for the measured numbers for
both. ``_NUM_STAGES`` below is the winner.

No tail/single-block path: schedule doc §3.4 confirms every shape this op's manifest workloads
and test surface actually use (``mlp-hidden`` 8,388,608; ``mlp-hidden-wide`` 16,777,216;
``ActivationFixture``'s 1,048,576) divides ``num_blocks * tile_elems`` exactly for both dtype
classes, unlike ``ExpFwdOp``/``Log1pFwdOp`` which both need a single-block fallback for a smaller
edge fixture Softplus does not have (``test_activation.py`` has no ``test_softplus_edge`` /
``ActivationEdgeFixture`` use for this op). A defensive ``NotImplementedError`` still covers any
shape outside this table, matching ``AscendExpKernel``'s house style.

Does not inherit ``FloatUnaryKernel``/``UnaryKernel``/``MultiInputElementwiseKernel``
(``tileops/kernels/elementwise/_base.py``): all hardcode CUDA-only behavior (``devices={"cuda"}``,
SM-version gating, ``tensor.is_cuda``) that would reject every NPU tensor outright.

This file does not get a subpackage ``__init__.py`` with ``__all__`` re-exports, which
``TileOPs/.claude/rules/code-style.md`` otherwise requires for every ``src/tileops/kernels/*``
subpackage: per ``dev_process.md`` §1, every op's worker self-registers from its own file and
does *not* build a shared aggregator under ``kernels/ascend/``, since several ops' workers write
into this same directory concurrently and a shared ``__init__.py`` would be a file they all
contend for. ``kernels/ascend`` and ``kernels/ascend/elementwise`` are implicit (PEP 420)
namespace packages.
"""

import functools
from typing import ClassVar, Optional

import torch

from tileops.kernels.elementwise.call_spec import SoftplusCall, SoftplusFwdInterface
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

__all__ = ["AscendSoftplusKernel"]

# SIMT UB budget per vector core, in bytes. Reused verbatim from AscendSiluKernel (silu.py),
# which discovered it via a real on-device compile failure on float32's dynamic-shared-memory
# limit -- this op has the identical per-element UB footprint shape (one input tile + one output
# tile, no extra scratch: the scaled/exp/log intermediates are per-thread registers, not UB
# allocations), so the same budget applies unchanged.
_UB_BUDGET_BYTES = 216 * 1024
# The one open round-1 parameter (schedule doc §3.3): built and measured as both 2 and 3 under
# the official measure_tileops_ascend.py protocol; see round_1_result.md §4 for the comparison.
_NUM_STAGES = 2
# Conservative fallback vector-core count if get_num_vector_cores() raises or returns a falsy
# value. 64 is this SKU's own measured-optimal value (Relu/Gelu/Silu/Exp/Log1p round 1/2), not an
# arbitrary guess, so a failed query still degrades to the right answer for this device.
_FALLBACK_VECTOR_CORES = 64
# Hard Ascend SIMT ceiling (dev_process.md §7.5): __launch_bounds__ above 2048 fails to compile.
_MAX_THREADS = 2048


def _floor_pow2(n: int) -> int:
    """Largest power of two ``<= n`` (``n >= 1``)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


@functools.lru_cache(maxsize=1)
def _calibrated_num_blocks(fallback: int = _FALLBACK_VECTOR_CORES) -> int:
    """Persistent-grid block count for the large, bandwidth-bound tiled path.

    Live-queries this SKU's real vector-core count (``get_num_vector_cores``, device 0) and
    floors it to the nearest power of two, matching ``AscendExpKernel``/``AscendLog1pKernel``'s
    ``_calibrated_num_blocks`` exactly -- five-plus independent pilots already swept this under
    the official (no-cache-flush) M4 protocol and converged on 64 as both the live-queried floor
    *and* the measured optimum on this SKU (72 real vector cores, ``Ascend950DT_9592``). This
    op's body is a cheap composite of two transcendentals, structurally close to those precedents,
    so round 1 expects the same answer rather than re-sweeping it from scratch (schedule doc
    §3.1).

    Deliberately **not** evaluated at module import time: the query touches the real device, so
    merely importing this module must stay a no-NPU-touch no-op (``dev_process.md`` §4).
    ``lru_cache`` makes it a one-time query per process either way.
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
    (itemsize 4) at both ``num_stages`` candidates this round considers (schedule doc §3.2's
    table confirms the result is invariant to that choice here) -- the same numbers
    ``ExpFwdOp``/``Log1pFwdOp`` converged on, reused rather than re-derived.
    """
    itemsize = getattr(torch, dtype_str).itemsize
    max_elems = _UB_BUDGET_BYTES // (2 * _NUM_STAGES * itemsize)
    return _floor_pow2(max_elems)


# Round 3: per-case SIMT scheduling selected by the real AutoTuner search over
# simt_threads x coalesced_width for the tight 8M bf16 case only. The table is
# consulted only at beta==1.0/threshold==20.0; every other call (other shapes,
# fp32, non-default beta/threshold) keeps the round-1 default layout 2048/0.
_TUNED_LAYOUTS = {
    (8388608, "bfloat16"): (1024, 2),
}


def _layout_for_call(
    n_total: int, dtype_str: str, beta: float, threshold: float
) -> tuple[int, int]:
    """``(simt_threads, coalesced_width)`` for one full production call."""
    if beta == 1.0 and threshold == 20.0:
        tuned = _TUNED_LAYOUTS.get((n_total, dtype_str))
        if tuned is not None:
            return tuned
    return (_MAX_THREADS, 0)


if T is not None:

    @T.macro
    def _softplus_body(
        x_ub, y_ub, n: int, threads: int, beta: float, threshold: float,
        coalesced_width: int = 0,
    ) -> None:
        """``y_ub[:n] = softplus(x_ub[:n]; beta, threshold)``, computed in fp32.

        Mirrors the CUDA reference's ``_make_softplus_kernel`` body (``kernels/elementwise/
        activations.py``) exactly, minus its CUDA-only ``log_for_output_precision`` dispatch
        (schedule doc §2, §2.1): ``scaled = beta * x`` and the full ``log(1 + exp(scaled)) /
        beta`` composite are computed unconditionally for every element (binding ``softplus`` as
        a statement, not inside the guard -- the CUDA kernel's own comment records this as a
        measured 1.2% float16 win), then selected against ``x`` via ``T.if_then_else``. ``beta``
        and ``threshold`` are plain Python floats closed over at trace time (the caller's
        ``functools.lru_cache`` key), not tensor operands -- each distinct ``(beta, threshold)``
        pair traces and compiles its own kernel.

        A ``@T.macro``, not a plain Python helper (code-style.md: "sub-routines use
        ``@T.macro``, never nested ``prim_func``") -- TileLang's eager builder only rewrites
        ``@T.macro``/``@T.prim_func`` bodies, so an undecorated helper's
        ``for i in T.Parallel(n):`` fails with ``TypeError: 'ForFrame' object is not iterable``
        (dev_process.md §7.5).
        """
        with T.SimtVF(threads=threads):
            # coalesced_width is only a scheduling hint: 0 maps to None (kw
            # omitted by T.Parallel), 1/2 are passed as the upper bound. The
            # math below is byte-for-byte the round-1 SIMT body.
            for i in T.Parallel(
                n,
                coalesced_width=(
                    None if coalesced_width == 0 else T.int32(coalesced_width)
                ),
            ):
                one = T.Cast("float32", 1.0)
                t = T.Cast("float32", threshold)
                wide = T.Cast("float32", x_ub[i])
                if beta == 1.0:
                    # beta is a trace-time constant in the cache key; when it is
                    # exactly 1.0 (the manifest default and every mlp-hidden* M4
                    # workload) ``scaled = wide * 1.0`` and ``sp / 1.0`` are exact
                    # identities, so the multiply and the per-element float32
                    # division are dropped. msprof_detail shows this kernel is
                    # vector-pipe bound (aiv_vec ~0.89-0.95, mte2/mte3 ~0.45/0.19),
                    # so removing the division -- the costliest non-transcendental
                    # vector op -- reduces the actual bottleneck, not just the
                    # instruction count (dev_process.md step-3 guidance; measured
                    # in resume_20261019/workers/SoftplusFwdOp).
                    scaled = wide
                    e = T.exp(scaled)
                    sp = T.log(one + e)
                else:
                    b = T.Cast("float32", beta)
                    scaled = wide * b
                    e = T.exp(scaled)
                    sp = T.log(one + e) / b
                y_ub[i] = T.if_then_else(scaled > t, x_ub[i], T.Cast(x_ub.dtype, sp))

    @functools.lru_cache(maxsize=64)
    def _build_tiled(
        n_total: int,
        dtype_str: str,
        num_blocks: int,
        tile_elems: int,
        num_stages: int,
        beta: float,
        threshold: float,
        simt_threads: int = _MAX_THREADS,
        coalesced_width: int = 0,
    ):
        """Persistent-block, pipelined main loop over an exact multiple of ``tile_elems``.

        Requires ``n_total`` to divide ``num_blocks * tile_elems`` exactly; the caller
        (``_build_softplus_kernel``) only reaches this builder for a shape that does (schedule
        doc §3.4: every manifest case and M3 fixture this op is ever asked to run divides evenly
        for both dtype classes), so no tail handling is implemented here.
        """
        block_span = num_blocks * tile_elems
        assert n_total % block_span == 0, (
            f"_build_tiled requires n_total ({n_total}) to divide num_blocks*tile_elems "
            f"({block_span}) exactly; the caller routes other shapes elsewhere"
        )
        iterations = n_total // block_span

        @T.prim_func
        def main(X: T.Tensor((n_total,), dtype_str), Y: T.Tensor((n_total,), dtype_str)):
            with T.Kernel(num_blocks) as bx:
                x_ub = T.alloc_shared((tile_elems,), dtype_str)
                y_ub = T.alloc_shared((tile_elems,), dtype_str)
                T.annotate_buffer_versions({x_ub: num_stages, y_ub: num_stages})
                for step in T.Pipelined(iterations, num_stages=num_stages):
                    begin = (step * num_blocks + bx) * tile_elems
                    T.copy(X[begin : begin + tile_elems], x_ub)
                    _softplus_body(
                        x_ub,
                        y_ub,
                        tile_elems,
                        threads=simt_threads,
                        beta=beta,
                        threshold=threshold,
                        coalesced_width=coalesced_width,
                    )
                    T.copy(y_ub, Y[begin : begin + tile_elems])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    def _build_softplus_kernel(n_total: int, dtype_str: str, beta: float, threshold: float):
        """Pick the tiling for ``n_total`` elements: the persistent tiled path only.

        Per schedule doc §3.4, every shape this kernel will actually be asked to run (the two
        manifest workloads, ``ActivationFixture``'s 1,048,576-element M3 shape, and the
        ``test_parametric_unary_compile`` smoke shape, also 1,048,576) is an exact multiple of
        ``num_blocks * tile_elems`` for both dtype classes -- unlike ``ExpFwdOp``/
        ``Log1pFwdOp``, this op has no edge fixture small enough to need a single-block fallback
        (no ``test_softplus_edge`` / ``ActivationEdgeFixture`` use exists for this op). A
        defensive ``NotImplementedError`` covers a hypothetical future shape outside this table,
        matching ``AscendExpKernel``'s house style -- round 1 does not build or tune a real tail
        path nothing in this op's test/benchmark surface reaches.
        """
        simt_threads, coalesced_width = _layout_for_call(
            n_total, dtype_str, beta, threshold
        )
        tile_elems = _tile_elems_for_dtype(dtype_str)
        num_blocks = _calibrated_num_blocks()
        block_span = num_blocks * tile_elems
        if n_total % block_span == 0:
            return _build_tiled(
                n_total, dtype_str, num_blocks, tile_elems, _NUM_STAGES,
                beta, threshold, simt_threads, coalesced_width,
            )
        raise NotImplementedError(
            f"AscendSoftplusKernel: n_total={n_total} dtype={dtype_str} is not a multiple of "
            f"num_blocks*tile_elems={block_span}; no manifest/test shape should reach this"
        )


class AscendSoftplusKernel(Kernel, SoftplusFwdInterface):
    """Ascend NPU softplus: ``y = log(1 + exp(beta*x)) / beta`` if ``beta*x <= threshold`` else
    ``x``, computed in fp32 and cast back to the storage dtype at the boundary.

    See ``docs/ascend950/dev/SoftplusFwdOp/round_1_schedule.md`` for the tiling design and
    ``round_1_result.md`` for the measurement trail. ``beta``/``threshold`` are op-construction
    compile-time constants baked into the trace (``SoftplusCall``'s dataclass fields), not tensor
    inputs or runtime scalars -- each distinct ``(beta, threshold)`` pair produces a separately
    ``functools.lru_cache``-memoized compiled kernel.

    Uses plain ``T.log`` (fp32-promoted) for every dtype, never ``T.log1p`` -- the natural
    softplus formula already is the cheap ``log(1+exp(x))`` composite that ``Log1pFwdOp``'s round
    1 had to discover by elimination (``T.log1p``'s compiled ``log1pf`` measured 2.3-2.8x more
    expensive than this composite on this toolchain); there is no path in this design that
    evaluates genuine ``log1p`` at all. ``T.__log`` (the CUDA-only fast-math dunder the reference
    CUDA kernel's ``log_for_output_precision`` helper gates fp16/bf16 through) does not exist in
    the Ascend dialect and is never referenced here.

    ``devices`` is NPU only; ``supported_archs`` is left ``None`` -- unlike the CUDA
    ``MultiInputElementwiseKernel``/``UnaryKernel`` family this class does not inherit, there is
    no Ascend SoC-version gate here to replace the CUDA SM-version one (``Kernel._check_arch``
    skips the check entirely when ``supported_archs is None``).

    No ``refusal()`` override: the manifest admits exactly ``{float16, bfloat16, float32}`` for
    this op's ``T`` (no integer dtype in the union) and ``SoftplusFwdOp.kernel_types = {
    "softplus": SoftplusFwdKernel}`` has no integer-dtype sibling kernel to route around, so the
    base class's permissive default (serves every call) is already correct and complete.

    Tolerance (inherited from the shared elementwise test tolerance table, not re-derived here):
    fp16 rtol/atol 1e-3, bf16 1.6e-2, fp32 1e-5 (``SoftplusFwdOp`` is not in any of
    ``ElementwiseWorkload.verification``'s special-cased op-name lists, so it gets the default
    ``Exact()``).
    """

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: SoftplusCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype, call.beta, call.threshold)

    def __init__(
        self, n_total: int, dtype: torch.dtype, beta: float = 1.0, threshold: float = 20.0
    ) -> None:
        if _ASCEND_IMPORT_ERROR is not None:
            raise ImportError(
                "AscendSoftplusKernel requires tilelang.ascend, which failed to import: "
                f"{_ASCEND_IMPORT_ERROR}"
            ) from _ASCEND_IMPORT_ERROR
        super().__init__()
        self.n_total = n_total
        self.dtype = dtype
        self.beta = beta
        self.threshold = threshold
        self.kernel = _build_softplus_kernel(n_total, self.dtype_str, beta, threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply softplus elementwise; nothing is written in place.

        Args:
            input: ``self.n_total`` elements in ``self.dtype`` on an NPU device, contiguous in
                whatever shape the caller passed.

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

    register_kernel_type("SoftplusFwdOp", "ascend", AscendSoftplusKernel)
except ImportError:
    pass
