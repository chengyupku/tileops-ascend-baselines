"""SGL's own implementations; importing this registry never imports torch/NPU."""
from .elementwise import sigmoid, silu, sub, neg, silu_and_mul
from .reduction import argmax, mean
from .norm import layernorm, fused_add_rms, groupnorm, instancenorm

PROVIDERS = {'SigmoidFwdOp': sigmoid, 'SiluFwdOp': silu, 'SubFwdOp': sub,
             'NegFwdOp': neg, 'SiluAndMulFwdOp': silu_and_mul, 'ArgmaxFwdOp': argmax,
             'MeanFwdOp': mean, 'LayerNormFwdOp': layernorm,
             'FusedAddRMSNormFwdOp': fused_add_rms, 'GroupNormFwdOp': groupnorm,
             'InstanceNormFwdOp': instancenorm}

from .matmul import bmm, grouped_gemm
from .cache import paged_cache_write
from .linear import gdn, kda, delta_recurrent, delta_chunk
from . import classics
PROVIDERS.update({'BmmFwdOp':bmm,'GroupedGemmFwdOp':grouped_gemm,
                  'PagedKVCacheWriteFwdOp':paged_cache_write,'GDNFwdOp':gdn,'KDAFwdOp':kda,
                  'DeltaNetRecurrentFwdOp':delta_recurrent,'DeltaNetChunkFwdOp':delta_chunk,
                  'AddFwdOp':classics.add,'AmaxFwdOp':classics.amax,'RMSNormFwdOp':classics.rms,
                  'GemmFwdOp':classics.gemm,'Conv2dFwdOp':classics.conv2d,'RoPEFwdOp':classics.rope,
                  'DeltaNetFwdOp':classics.delta,'GQADenseFwdOp':classics.gqa})
from .rope import rope_variant
from .attention import varlen, paged
PROVIDERS.update({op:rope_variant for op in ('RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp','RoPENeoxPositionIdsFwdOp')})
PROVIDERS.update({'GQAVarlenFwdOp':varlen,'GQAPagedFwdOp':paged})
from .sampling import chain, categorical
PROVIDERS.update({'ChainSpeculativeSamplingFwdOp':chain,'SamplingFromProbsFwdOp':categorical})
from .additional import mul, maximum, fused_add_layernorm, ada_layernorm, mean_pooling
PROVIDERS.update({'MulFwdOp':mul,'MaximumFwdOp':maximum,'FusedAddLayerNormFwdOp':fused_add_layernorm,
                  'AdaLayerNormFwdOp':ada_layernorm,'MeanPoolingFwdOp':mean_pooling})
from .attention import mla_decode, mha_paged
PROVIDERS.update({'MLADecodeWithKVCacheFwdOp':mla_decode,'MHADecodePagedWithKVCacheFwdOp':mha_paged})
from .moe import grouped, expert_mlp
PROVIDERS.update({'MoEGroupedGemmFwdOp':grouped,'MoEExpertMLPFwdOp':expert_mlp})
from .additional import ada_zero, minimum
from .selection_reduce import argmin, amin
PROVIDERS.update({'AdaLayerNormZeroFwdOp':ada_zero,'MinimumFwdOp':minimum,'ArgminFwdOp':argmin,'AminFwdOp':amin})
from .attention import dsa_decode
PROVIDERS['DSADecodeWithKVCacheFwdOp']=dsa_decode

from .pooling import pool
PROVIDERS.update({op:pool for op in ('AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp','MaxPool1dFwdOp','MaxPool1dIndicesFwdOp','MaxPool2dFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dFwdOp','MaxPool3dIndicesFwdOp','AdaptiveAvgPool2dFwdOp','AdaptiveMaxPool2dFwdOp','AdaptiveMaxPool2dIndicesFwdOp')})

from .scalar import selectors, lerp_scalar, sum_reduce
PROVIDERS.update({op:selectors for op in ('AbsFwdOp','ReluFwdOp','ClampScalarFwdOp','ClampTensorFwdOp','HardtanhFwdOp')})
PROVIDERS.update({'LerpScalarFwdOp':lerp_scalar,'SumFwdOp':sum_reduce})
from .attention import prefill_append
PROVIDERS['GQAPrefillPagedWithKVCacheFwdOp']=prefill_append
from .convolution import conv_nd
PROVIDERS.update({'Conv1dFwdOp':conv_nd,'Conv3dFwdOp':conv_nd})
from .indexer import topk_select
PROVIDERS['TopKSelectFwdOp']=topk_select
from .attention import nsa_selected
PROVIDERS['NSAVarlenFwdOp']=nsa_selected
from .batchnorm import batchnorm
PROVIDERS['BatchNormFwdOp']=batchnorm
from .convolution import conv2d as full_window_conv2d
PROVIDERS['Conv2dFwdOp']=full_window_conv2d
from .norm_compat import fused_add_rms as compatible_fused_add_rms, instancenorm as complete_instancenorm
PROVIDERS.update({'FusedAddRMSNormFwdOp':compatible_fused_add_rms,'InstanceNormFwdOp':complete_instancenorm})
from .composed import activation as composed_activation, variance
PROVIDERS.update({op:composed_activation for op in ('HardsigmoidFwdOp','HardswishFwdOp','LeakyReluFwdOp','PreluFwdOp')})
PROVIDERS.update({'VarFwdOp':variance,'VarMeanFwdOp':variance})
from .mean_compat import mean as compatible_mean
PROVIDERS['MeanFwdOp']=compatible_mean
from .conditional import select
PROVIDERS.update({op:select for op in ('WhereFwdOp','MaskedFillScalarFwdOp','MaskedFillTensorFwdOp')})
from .extended import causal_conv,qk_norm_rope
PROVIDERS.update({'CausalConv1dFwdOp':causal_conv,'CausalConv1dUpdateFwdOp':causal_conv,'FusedQKNormRoPEFwdOp':qk_norm_rope})
from .mla_cache import write as mla_cache_write
PROVIDERS['MLAKVCacheWriteFwdOp']=mla_cache_write
from .topk_mask import build as topk_mask
PROVIDERS['TopKMaskFwdOp']=topk_mask
from .gather import build as paged_gather
PROVIDERS['PagedKVCacheGatherFwdOp']=paged_gather
from .softplus import build as softplus
PROVIDERS['SoftplusFwdOp']=softplus
from .extra_arithmetic import dequant, tanh, lerp
PROVIDERS.update({op:dequant for op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp')})
PROVIDERS.update({'TanhFwdOp':tanh,'LerpTensorFwdOp':lerp})
from .chunk_complete import build as complete_delta_chunk
PROVIDERS['DeltaNetChunkFwdOp'] = complete_delta_chunk
from .extra_arithmetic import lerp_scalar as complete_lerp_scalar
PROVIDERS['LerpScalarFwdOp'] = complete_lerp_scalar
from .row_gemv import mean as complete_row_mean, sum_reduce as complete_row_sum
PROVIDERS.update({'MeanFwdOp':complete_row_mean,'SumFwdOp':complete_row_sum})
from .boolean_reduce import build as boolean_reduce
PROVIDERS.update({op:boolean_reduce for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp')})
from .vector_norm import build as vector_norm
PROVIDERS['VectorNormFwdOp'] = vector_norm
from .boolean_elementwise import build as boolean_elementwise
PROVIDERS.update({op:boolean_elementwise for op in ('BitwiseAndFwdOp','BitwiseOrFwdOp','BitwiseXorFwdOp','LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp')})
from .quant_gemv import build as quant_gemv
PROVIDERS.update({op:quant_gemv for op in ('BmmFP8FwdOp','GemmFP8FwdOp','GemmINT8W8A8FwdOp')})
from .comparison import build as comparison
PROVIDERS.update({op:comparison for op in ('EqFwdOp','NeFwdOp','GeFwdOp','GtFwdOp','LeFwdOp','LtFwdOp')})
from .finite_truth import build as finite_truth
PROVIDERS.update({op:finite_truth for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp','LogicalAndFwdOp','LogicalOrFwdOp','LogicalNotFwdOp')})
from .lightning_logits import build as lightning_logits
PROVIDERS['FP8LightningIndexerFwdOp'] = lightning_logits
from .exp_composed import build as exp_composed
PROVIDERS.update({op:exp_composed for op in ('ExpFwdOp','Expm1FwdOp','EluFwdOp','SeluFwdOp')})
from .sign import build as sign
PROVIDERS['SignFwdOp'] = sign
from .inverse_rms import build as positive_inverse
PROVIDERS.update({op:positive_inverse for op in ('ReciprocalFwdOp','DivFwdOp')})
from .std import build as std
PROVIDERS['StdFwdOp'] = std
from .mish import build as mish
PROVIDERS['MishFwdOp'] = mish
from .alibi import build as alibi
PROVIDERS['AlibiFwdOp'] = alibi
from .bitwise_not import build as bitwise_not
PROVIDERS['BitwiseNotFwdOp'] = bitwise_not
from .probability_composed import build as probability_composed
PROVIDERS.update({op:probability_composed for op in ('SoftmaxFwdOp','MinPMaskFwdOp')})
from .w4a16 import build as w4a16
PROVIDERS['GemmW4A16FwdOp'] = w4a16
from .moe_post import build as moe_post
PROVIDERS['MoEPostPermuteFwdOp'] = moe_post
from .gelu_tanh_mul import build as gelu_tanh_mul
PROVIDERS['GeluTanhAndMulFwdOp'] = gelu_tanh_mul
from .dropout import build as dropout
PROVIDERS['DropoutFwdOp'] = dropout
from .mhc_post import build as mhc_post
PROVIDERS['MHCPostFwdOp'] = mhc_post
from .extended_truth import build as extended_truth
PROVIDERS.update({op:extended_truth for op in ('AnyFwdOp','AllFwdOp','CountNonzeroFwdOp')})
from .ssd_components import cumsum as ssd_cumsum, coupling as ssd_coupling
PROVIDERS.update({'SSDChunkCumsumFwdOp':ssd_cumsum,'SSDChunkCouplingFwdOp':ssd_coupling})

from .integration700_sgl_recovery01 import bmm as _recovery_bmm, paged_gather as _recovery_paged_gather
PROVIDERS.update({
    'BmmFwdOp': _recovery_bmm, 'PagedKVCacheGatherFwdOp': _recovery_paged_gather,
})

from .integration700_sgl_recovery01 import gdn as _recovery_gdn
PROVIDERS["GDNFwdOp"] = _recovery_gdn
