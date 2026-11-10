"""DeepGEMM-Ascend all-pairs providers. Importing this package is CPU-safe.

Factories require a provenance-checked local build and an externally authorized
device reservation. No factory or kernel was executed in the CPU-only audit.
"""

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from benchmarks.ascend_baselines import Candidate

from .adapters import (
    bmm,
    convolution,
    fp8_bmm,
    fp8_gemm,
    gemm,
    grouped_gemm,
    moe_grouped_gemm,
    ssd_chunk_coupling,
)

PROVIDERS: dict[str, Callable[[Any], "Candidate"]] = {
    "GemmFwdOp": gemm,
    "BmmFwdOp": bmm,
    "GroupedGemmFwdOp": grouped_gemm,
    "GemmFP8FwdOp": fp8_gemm,
    "BmmFP8FwdOp": fp8_bmm,
    "Conv1dFwdOp": convolution,
    "Conv2dFwdOp": convolution,
    "Conv3dFwdOp": convolution,
    "MoEGroupedGemmFwdOp": moe_grouped_gemm,
    "SSDChunkCouplingFwdOp": ssd_chunk_coupling,
}
