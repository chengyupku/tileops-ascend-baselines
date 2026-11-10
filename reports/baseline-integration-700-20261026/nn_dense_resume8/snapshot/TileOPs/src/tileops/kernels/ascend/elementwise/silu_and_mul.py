"""TileLang-Ascend kernel for ``SiluAndMulFwdOp``: y = silu(gate) * value, silu in fp32.

New, additive file: it does not edit any existing kernel/op module. It self-registers
with ``tileops.backend.register_kernel_type`` at import time (see
``docs/ascend950/dev_process.md`` section 1), so importing this module is what makes
``AscendSiluAndMulKernel`` participate in ``SiluAndMulFwdOp`` dispatch for npu tensors;
nothing else needs to change.

The input ``x`` is a single ``(M, 2*N)`` row-major tensor; ``gate = x[:, :N]`` and
``value = x[:, N:]``, and the output is ``silu(gate) * value`` with shape ``(M, N)``
(``FusedGatedFwdInterface``, ``tileops/kernels/elementwise/call_spec.py``).

Structurally this is the batch-1 ``GeluAndMulFwdOp`` kernel
(``tileops/kernels/ascend/elementwise/gelu_and_mul.py``) with its erf polynomial swapped
for ``silu(g) = g / (1 + exp(-g))`` -- the same fp32 formula
``tileops/kernels/ascend/elementwise/silu.py``'s ``_silu_op`` already validated on this
hardware for the non-fused ``SiluFwdOp`` (not reused by import: that module's helper is
private, and round_1_schedule.md calls for porting the math only). The CUDA reference
kernel (``tileops/kernels/elementwise/activations.py``'s ``SiluAndMulFwdKernel``) instead
uses ``exp2(-gate*log2e)`` plus a ``float32_quotient`` range-checked divide -- both are
CUDA-specific: ``exp2`` targets the ``MUFU.EX2`` instruction and ``float32_quotient``
works around ``__fdividef``'s fast-path precision loss, neither of which has an Ascend
analog or an Ascend-specific reason to be ported (round_1_schedule.md section 6).
"""

import functools
from typing import ClassVar

import torch

try:
    import tilelang
    import tilelang.ascend.language as T

    from tileops.backend import register_kernel_type
    from tileops.kernels.elementwise.call_spec import FusedGatedCall, FusedGatedFwdInterface
    from tileops.kernels.kernel_base import Entry, Kernel

    _ASCEND_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without an Ascend env
    _ASCEND_IMPORT_ERROR = exc

__all__ = ["AscendSiluAndMulKernel"] if _ASCEND_IMPORT_ERROR is None else []


if _ASCEND_IMPORT_ERROR is None:
    _TORCH_DTYPE_TO_TL = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }

    # Ascend 950 AIV core count this round's tiling assumes. Not independently queried from
    # the device; deliberately kept at 64 rather than the "real" 72 cube/vector cores
    # dev_process.md section 7.5 generally recommends checking: SiluFwdOp round 2 already
    # swept {56, 64, 112} plus a genuine two-launch 72-core split on this exact N=14336
    # dimension and found 64 fastest in both comparisons (round_1_schedule.md section 4.1),
    # and all three of this op's N values (14336, 14336, 12288) are 2^a * {7, 7, 3} with no
    # factor of 3^2, so 72 = 2^3*3^2 cannot divide any of them evenly anyway.
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

    def _silu_and_mul_program(m: int, n: int, dtype: str):
        """Build the ``(m, 2*n) -> (m, n)`` SiluAndMul prim_func for one (m, n, dtype)."""
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
                            # Promote to float32 for the exp/divide (code-style.md:
                            # overflow-prone fp16/bf16 math goes to fp32), cast back to the
                            # storage dtype before the (cheap, not overflow-prone) multiply
                            # by value -- same boundary gelu_and_mul.py/silu.py use.
                            g = T.Cast("float32", gate_ub[i])
                            one = T.cast(1.0, "float32")
                            silu_val = g / (one + T.exp(-g))
                            out_ub[i] = T.Cast(dtype, silu_val) * value_ub[i]
                    T.copy(out_ub, Y[row, col : col + block_n])

        return main

    @functools.lru_cache(maxsize=None)
    def _build_silu_and_mul_kernel(m: int, n: int, dtype_str: str):
        program = _silu_and_mul_program(m, n, dtype_str)
        return tilelang.compile(program, target="ascend", out_idx=-1)

    class AscendSiluAndMulKernel(Kernel, FusedGatedFwdInterface):
        """TileLang-Ascend SiluAndMul: ``y = silu(gate) * value``, ``silu`` in fp32.

        ``silu(g) = g / (1 + exp(-g))``, promoted to float32 and cast back to the storage
        dtype at the boundary, mirroring ``AscendSiluKernel._silu_op``
        (``tileops/kernels/ascend/elementwise/silu.py``) and the in-tree CUDA
        ``SiluAndMulFwdKernel.activation_func``'s algebra (not its CUDA-specific
        ``exp2``/``float32_quotient`` instruction choices -- see module docstring).
        """

        devices: ClassVar[frozenset[str]] = frozenset({"npu"})

        @classmethod
        def entry_for(cls, call: FusedGatedCall) -> Entry:
            return call, lambda: cls(call.m, call.n, call.dtype)

        def __init__(self, m: int, n: int, dtype: torch.dtype):
            super().__init__()
            if dtype not in _TORCH_DTYPE_TO_TL:
                raise ValueError(f"AscendSiluAndMulKernel does not support dtype {dtype!r}")
            self.m = m
            self.n = n
            self.dtype = dtype
            self._dtype_str = _TORCH_DTYPE_TO_TL[dtype]
            self.kernel = _build_silu_and_mul_kernel(m, n, self._dtype_str)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``silu(x[:, :n]) * x[:, n:]``; nothing is written in place."""
            if not x.is_contiguous():
                x = x.contiguous()
            return self.kernel(x)

    register_kernel_type("SiluAndMulFwdOp", "ascend", AscendSiluAndMulKernel)
