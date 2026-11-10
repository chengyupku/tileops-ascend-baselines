"""TileLang-Ascend kernel for ``GeluAndMulFwdOp``: y = gelu(gate) * value, exact erf GELU.

New, additive file: it does not edit any existing kernel/op module. It self-registers
with ``tileops.backend.register_kernel_type`` at import time (see
``docs/ascend950/dev_process.md`` section 1), so importing this module is what makes
``AscendGeluAndMulKernel`` participate in ``GeluAndMulFwdOp`` dispatch for npu tensors;
nothing else needs to change.

The input ``x`` is a single ``(M, 2*N)`` row-major tensor; ``gate = x[:, :N]`` and
``value = x[:, N:]``, and the output is ``gelu(gate) * value`` with shape ``(M, N)``
(``FusedGatedFwdInterface``, ``tileops/kernels/elementwise/call_spec.py``).

The erf used on fp16/bf16 is the same clamped degree-10 polynomial the CUDA reference
kernel uses (``tileops/kernels/elementwise/_erf.py``'s ``gelu()``), ported as math only
(the constants and formula, not the code): that file imports ``tilelang.language``, the
CUDA dialect, which this module must not use. Ascend's SIMD MicroAPI exposes no
``verf``/``vtanh`` instruction, so avoiding a transcendental call entirely is also a
correctness requirement here, not just a performance choice.
"""

import functools
from typing import ClassVar

import torch

try:
    import tilelang
    import tilelang.ascend.language as T

    from tileops.backend import register_kernel_type
    from tileops.kernels.constants import INV_SQRT2
    from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
    from tileops.kernels.kernel_base import Entry, Kernel

    _ASCEND_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without an Ascend env
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendGeluAndMulKernel"] if _ASCEND_IMPORT_ERROR is None else []


if _ASCEND_IMPORT_ERROR is None:
    # erf(x) = clip(t * P(w), -1, 1), t = clamp(x, -_ERF_CLAMP, _ERF_CLAMP),
    # w = 1 - (t / _ERF_CLAMP)**2, highest degree first. Ported verbatim (values only) from
    # tileops/kernels/elementwise/_erf.py: worst case over the real line is 1.7e-5, an order
    # below half a float16 ulp at 1.0. Ascend behavior of this exact polynomial has not been
    # measured before this round; M3 is what confirms it still holds under Ascend's FMA/
    # rounding.
    _ERF_CLAMP = 3.6
    _ERF_POLY_COEFFS = (
        8.971932411193848,
        -33.1397590637207,
        53.60123062133789,
        -48.02798080444336,
        26.02545738220215,
        -8.488715171813965,
        1.7499406337738037,
        -0.09359221160411835,
        0.11330155283212662,
        0.1387363225221634,
        0.2777777910232544,
    )

    _TORCH_DTYPE_TO_TL = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }

    # Ascend 950 AIV core count this round's tiling assumes (same value used by the other
    # batch-1 Ascend examples: example_rmsnorm.py, example_per_token_cast_to_fp8.py). Not
    # independently queried from the device; if measurement shows a different effective
    # core count this constant is the first thing to revisit.
    _NUM_AIV_CORES = 64

    def _largest_divisor_leq(n: int, upper_bound: int) -> int:
        """The largest divisor of *n* that is <= *upper_bound* (upper_bound clamped to >= 1)."""
        upper_bound = max(1, min(upper_bound, n))
        for candidate in range(upper_bound, 0, -1):
            if n % candidate == 0:
                return candidate
        return 1  # unreachable: 1 always divides n

    def _pick_block_n(m: int, n: int) -> int:
        """The column tile width.

        When rows alone give at least one tile per core (``m >= _NUM_AIV_CORES``, the
        prefill-shaped case), tile wide for DMA efficiency but cap at 8192 elements so
        double-buffered gate/value/output UB stays a comfortable fraction of the per-core
        budget. When rows do not (``m < _NUM_AIV_CORES``, the decode-shaped case, including
        ``m == 1``), parallelism can only come from splitting N, so target one tile per
        core instead.
        """
        if n <= 0:
            raise ValueError(f"N must be positive, got {n}")
        if m >= _NUM_AIV_CORES:
            return _largest_divisor_leq(n, 8192)
        return _largest_divisor_leq(n, max(1, n // _NUM_AIV_CORES))

    def _gelu_and_mul_program(m: int, n: int, dtype: str):
        """Build the ``(m, 2*n) -> (m, n)`` GeluAndMul prim_func for one (m, n, dtype)."""
        block_n = _pick_block_n(m, n)
        col_tiles = n // block_n
        total_tiles = m * col_tiles
        # Only worth double-buffering when a core actually revisits the pipeline (more than
        # one tile per core); a single pass per core (e.g. the decode case, where
        # total_tiles == _NUM_AIV_CORES) has nothing to overlap with.
        num_stages = 2 if total_tiles // _NUM_AIV_CORES >= 2 else 1
        threads = min(1024, block_n)

        @T.prim_func
        def main(X: T.Tensor((m, 2 * n), dtype), Y: T.Tensor((m, n), dtype)):
            with T.Kernel(_NUM_AIV_CORES) as core_id:
                for tile in T.Persistent([total_tiles], _NUM_AIV_CORES, core_id, num_stages=num_stages):
                    row = tile // col_tiles
                    col = (tile % col_tiles) * block_n
                    gate_ub = T.alloc_shared((block_n,), dtype)
                    value_ub = T.alloc_shared((block_n,), dtype)
                    out_ub = T.alloc_shared((block_n,), dtype)
                    T.annotate_buffer_versions(
                        {gate_ub: num_stages, value_ub: num_stages, out_ub: num_stages}
                    )
                    T.copy(X[row, col : col + block_n], gate_ub)
                    T.copy(X[row, n + col : n + col + block_n], value_ub)
                    with T.SimtVF(threads=threads):
                        for i in T.Parallel(block_n):
                            # Promote to float32 for the polynomial (code-style.md:
                            # overflow-prone fp16/bf16 math goes to fp32), cast back to the
                            # storage dtype before the (cheap, not overflow-prone) multiply
                            # by value -- same boundary the CUDA reference kernel uses.
                            g = T.Cast("float32", gate_ub[i])
                            half_x = T.cast(0.5, "float32") * g
                            scaled = g * T.cast(INV_SQRT2 / _ERF_CLAMP, "float32")
                            s = T.min(
                                T.max(scaled, T.cast(-1.0, "float32")), T.cast(1.0, "float32")
                            )
                            w = T.cast(1.0, "float32") - s * s
                            acc = T.cast(_ERF_POLY_COEFFS[0] * _ERF_CLAMP, "float32")
                            for coeff in _ERF_POLY_COEFFS[1:]:
                                acc = acc * w + T.cast(coeff * _ERF_CLAMP, "float32")
                            one_plus_erf = T.max(
                                s * acc + T.cast(1.0, "float32"), T.cast(0.0, "float32")
                            )
                            gelu_val = half_x * one_plus_erf
                            out_ub[i] = T.Cast(dtype, gelu_val) * value_ub[i]
                    T.copy(out_ub, Y[row, col : col + block_n])

        return main

    @functools.lru_cache(maxsize=None)
    def _build_gelu_and_mul_kernel(m: int, n: int, dtype_str: str):
        program = _gelu_and_mul_program(m, n, dtype_str)
        return tilelang.compile(program, target="ascend", out_idx=-1)

    class AscendGeluAndMulKernel(Kernel, FusedGatedFwdInterface):
        """TileLang-Ascend GeluAndMul: ``y = gelu(gate) * value`` (exact erf GELU).

        On float16 and bfloat16 the error function is evaluated with the same clamped
        degree-10 polynomial as the in-tree CUDA kernel
        (``tileops/kernels/elementwise/_erf.py``), promoted to float32 and cast back to
        the storage dtype at the boundary; its worst case over the real line is 1.7e-5.
        float32 reuses the same polynomial path rather than a true ``erf`` call -- no
        manifest case for this op exercises float32, and Ascend's compilability of the
        TVM ``erf``/``tanh`` intrinsics is unverified (see round_1_schedule.md section 7).
        """

        devices: ClassVar[frozenset[str]] = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: FusedGatedCall) -> Entry:
            return call, lambda: cls(call.m, call.n, call.dtype)

        def __init__(self, m: int, n: int, dtype: torch.dtype):
            super().__init__()
            if dtype not in _TORCH_DTYPE_TO_TL:
                raise ValueError(f"AscendGeluAndMulKernel does not support dtype {dtype!r}")
            self.m = m
            self.n = n
            self.dtype = dtype
            self._dtype_str = _TORCH_DTYPE_TO_TL[dtype]
            self.kernel = _build_gelu_and_mul_kernel(m, n, self._dtype_str)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``gelu(x[:, :n]) * x[:, n:]``; nothing is written in place."""
            if not x.is_contiguous():
                x = x.contiguous()
            return self.kernel(x)

    register_kernel_type("GeluAndMulFwdOp", "ascend", AscendGeluAndMulKernel)
