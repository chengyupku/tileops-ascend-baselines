"""Lazy providers for ops-tensor; importing this package never initializes NPU."""
from .adapters import (
    add, bmm, convolution, gemm, grouped_gemm, moe_grouped_gemm,
    max_pool, ssd_chunk_coupling, trunc,
)
from .reductions import avg_pool, mean_pooling, mean_reduction, sum_reduction

PROVIDERS = {
    "AddFwdOp": add,
    "TruncFwdOp": trunc,
    "GemmFwdOp": gemm,
    "BmmFwdOp": bmm,
    "GroupedGemmFwdOp": grouped_gemm,
    "MoEGroupedGemmFwdOp": moe_grouped_gemm,
    "Conv1dFwdOp": convolution,
    "Conv2dFwdOp": convolution,
    "Conv3dFwdOp": convolution,
    "SSDChunkCouplingFwdOp": ssd_chunk_coupling,
    "MaxPool1dFwdOp": max_pool,
    "MaxPool2dFwdOp": max_pool,
    "MaxPool3dFwdOp": max_pool,
    "AdaptiveMaxPool2dFwdOp": max_pool,
    "SumFwdOp": sum_reduction,
    "MeanFwdOp": mean_reduction,
    "MeanPoolingFwdOp": mean_pooling,
    "AvgPool1dFwdOp": avg_pool,
    "AvgPool2dFwdOp": avg_pool,
    "AvgPool3dFwdOp": avg_pool,
    "AdaptiveAvgPool2dFwdOp": avg_pool,
}

# integration700 transformer_attention_misc: full original grouped host entry.
from .integration700_transformer_attention_misc_grouped import factory as _i700_misc_grouped
def _i700_misc_grouped_route(case, previous=PROVIDERS.get('MoEGroupedGemmFwdOp')):
    if case.id == 'ds-v3-decode-mlp-up-float16':
        return _i700_misc_grouped(case)
    if previous is not None:
        return previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No registered factory for this exact key')
PROVIDERS['MoEGroupedGemmFwdOp'] = _i700_misc_grouped_route
