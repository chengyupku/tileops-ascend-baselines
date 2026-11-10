"""Isolated ops-transformer providers; importing this module is device-free."""
from .adapters import bmm

PROVIDERS = {'BmmFwdOp': bmm}

# integration700_transformer_dense: numerically verified exact key only.
from .integration700_transformer_dense_gelu import gelu as _integration700_transformer_dense_gelu
_integration700_transformer_dense_old_gelu = PROVIDERS.get('GeluTanhAndMulFwdOp')
def _integration700_transformer_dense_gelu_dispatch(case):
    if case.id == 'llama-8b-ffn-prefill-float16':
        return _integration700_transformer_dense_gelu(case)
    if _integration700_transformer_dense_old_gelu is not None:
        return _integration700_transformer_dense_old_gelu(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No integration700 transformer_dense registration for this exact key')
PROVIDERS['GeluTanhAndMulFwdOp'] = _integration700_transformer_dense_gelu_dispatch

# integration700_transformer_dense: numerically verified exact YaRN key only.
from .integration700_transformer_dense_rope import rope as _integration700_transformer_dense_rope
_integration700_transformer_dense_old_yarn = PROVIDERS.get('YaRNFwdOp')
def _integration700_transformer_dense_rope_dispatch(case):
    if case.id == '1d-5k-float32':
        return _integration700_transformer_dense_rope(case)
    if _integration700_transformer_dense_old_yarn is not None:
        return _integration700_transformer_dense_old_yarn(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No integration700 transformer_dense registration for this exact key')
PROVIDERS['YaRNFwdOp'] = _integration700_transformer_dense_rope_dispatch

# integration700_transformer_dense graph exact verified keys
from .integration700_transformer_dense_rope import rope as _integration700_transformer_dense_LongRoPEFwdOp_factory
_integration700_transformer_dense_LongRoPEFwdOp_old = PROVIDERS.get('LongRoPEFwdOp')
def _integration700_transformer_dense_LongRoPEFwdOp_dispatch(case):
    if case.id in ['1d-5k-float32']: return _integration700_transformer_dense_LongRoPEFwdOp_factory(case)
    if _integration700_transformer_dense_LongRoPEFwdOp_old is not None: return _integration700_transformer_dense_LongRoPEFwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['LongRoPEFwdOp'] = _integration700_transformer_dense_LongRoPEFwdOp_dispatch
from .integration700_transformer_dense_topk import topk as _integration700_transformer_dense_FusedTopKFwdOp_factory
_integration700_transformer_dense_FusedTopKFwdOp_old = PROVIDERS.get('FusedTopKFwdOp')
def _integration700_transformer_dense_FusedTopKFwdOp_dispatch(case):
    if case.id in ['qwen3-235b-t32-bfloat16', 'qwen3-235b-t512-bfloat16']: return _integration700_transformer_dense_FusedTopKFwdOp_factory(case)
    if _integration700_transformer_dense_FusedTopKFwdOp_old is not None: return _integration700_transformer_dense_FusedTopKFwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['FusedTopKFwdOp'] = _integration700_transformer_dense_FusedTopKFwdOp_dispatch

# integration700_transformer_dense graph exact verified keys
from .integration700_transformer_dense_post import post as _integration700_transformer_dense_MoEPostPermuteFwdOp_factory
_integration700_transformer_dense_MoEPostPermuteFwdOp_old = PROVIDERS.get('MoEPostPermuteFwdOp')
def _integration700_transformer_dense_MoEPostPermuteFwdOp_dispatch(case):
    if case.id in ['decode-t32-float16']: return _integration700_transformer_dense_MoEPostPermuteFwdOp_factory(case)
    if _integration700_transformer_dense_MoEPostPermuteFwdOp_old is not None: return _integration700_transformer_dense_MoEPostPermuteFwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['MoEPostPermuteFwdOp'] = _integration700_transformer_dense_MoEPostPermuteFwdOp_dispatch

# integration700_transformer_dense graph exact verified keys
from .integration700_transformer_dense_quant import quant as _integration700_transformer_dense_GemmINT8W8A8FwdOp_factory
_integration700_transformer_dense_GemmINT8W8A8FwdOp_old = PROVIDERS.get('GemmINT8W8A8FwdOp')
def _integration700_transformer_dense_GemmINT8W8A8FwdOp_dispatch(case):
    if case.id in ['llama3-70b-dec-qkv-tsr-bfloat16']: return _integration700_transformer_dense_GemmINT8W8A8FwdOp_factory(case)
    if _integration700_transformer_dense_GemmINT8W8A8FwdOp_old is not None: return _integration700_transformer_dense_GemmINT8W8A8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['GemmINT8W8A8FwdOp'] = _integration700_transformer_dense_GemmINT8W8A8FwdOp_dispatch

# integration700_transformer_dense graph exact verified keys
from .integration700_transformer_dense_pre import pre as _integration700_transformer_dense_MoEPrePermuteFwdOp_factory
_integration700_transformer_dense_MoEPrePermuteFwdOp_old = PROVIDERS.get('MoEPrePermuteFwdOp')
def _integration700_transformer_dense_MoEPrePermuteFwdOp_dispatch(case):
    if case.id in ['decode-t32-bfloat16', 'decode-t32-float16']: return _integration700_transformer_dense_MoEPrePermuteFwdOp_factory(case)
    if _integration700_transformer_dense_MoEPrePermuteFwdOp_old is not None: return _integration700_transformer_dense_MoEPrePermuteFwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['MoEPrePermuteFwdOp'] = _integration700_transformer_dense_MoEPrePermuteFwdOp_dispatch

# integration700_transformer_dense graph exact verified keys
from .integration700_transformer_dense_fp8_bmm import fp8_bmm as _integration700_transformer_dense_BmmFP8FwdOp_factory
_integration700_transformer_dense_BmmFP8FwdOp_old = PROVIDERS.get('BmmFP8FwdOp')
def _integration700_transformer_dense_BmmFP8FwdOp_dispatch(case):
    if case.id in ['square-1k-float8_e4m3fn-bfloat16']: return _integration700_transformer_dense_BmmFP8FwdOp_factory(case)
    if _integration700_transformer_dense_BmmFP8FwdOp_old is not None: return _integration700_transformer_dense_BmmFP8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['BmmFP8FwdOp'] = _integration700_transformer_dense_BmmFP8FwdOp_dispatch
from .integration700_transformer_dense_fp8_moe import fp8_moe as _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_factory
_integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_old = PROVIDERS.get('MoEGroupedGemmFP8FwdOp')
def _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_dispatch(case):
    if case.id in ['qwen3-235b-dec-b64-down']: return _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_factory(case)
    if _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_old is not None: return _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['MoEGroupedGemmFP8FwdOp'] = _integration700_transformer_dense_MoEGroupedGemmFP8FwdOp_dispatch

# integration700_transformer_dense verified siblings 01
from .integration700_transformer_dense_quant import quant as _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_factory
_integration700_transformer_dense_01_GemmINT8W8A8FwdOp_old = PROVIDERS.get('GemmINT8W8A8FwdOp')
def _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_dispatch(case):
    if case.id in ['llama3-8b-decode-down-bfloat16', 'llama3-8b-decode-qkv-bfloat16', 'llama3-70b-prefill-down-bfloat16']: return _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_factory(case)
    if _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_old is not None: return _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['GemmINT8W8A8FwdOp'] = _integration700_transformer_dense_01_GemmINT8W8A8FwdOp_dispatch

# integration700_transformer_dense verified siblings 02
from .integration700_transformer_dense_fp8_bmm import fp8_bmm as _integration700_transformer_dense_02_BmmFP8FwdOp_factory
_integration700_transformer_dense_02_BmmFP8FwdOp_old = PROVIDERS.get('BmmFP8FwdOp')
def _integration700_transformer_dense_02_BmmFP8FwdOp_dispatch(case):
    if case.id in ['mha-decode-pv-float8_e4m3fn-bfloat16', 'mha-decode-pv-nk-float8_e4m3fn-bfloat16', 'mha-decode-qk-float8_e4m3fn-bfloat16', 'mha-decode-qk-nk-float8_e4m3fn-bfloat16', 'square-1k-nk-float8_e4m3fn-bfloat16', 'square-2k-float8_e4m3fn-bfloat16', 'square-2k-nk-float8_e4m3fn-bfloat16', 'moe-prefill-float8_e4m3fn-bfloat16', 'moe-prefill-nk-float8_e4m3fn-bfloat16']: return _integration700_transformer_dense_02_BmmFP8FwdOp_factory(case)
    if _integration700_transformer_dense_02_BmmFP8FwdOp_old is not None: return _integration700_transformer_dense_02_BmmFP8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['BmmFP8FwdOp'] = _integration700_transformer_dense_02_BmmFP8FwdOp_dispatch
from .integration700_transformer_dense_fp8_moe import fp8_moe as _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_factory
_integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_old = PROVIDERS.get('MoEGroupedGemmFP8FwdOp')
def _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_dispatch(case):
    if case.id in ['ds-v3-dec-b256-down', 'ds-v3-dec-b256-up', 'ds-v3-dec-b64-down', 'ds-v3-dec-b64-up', 'ds-v3-e8-down-masked', 'ds-v3-e8-up-masked', 'kimi-k2-dec-b256-down', 'kimi-k2-dec-b256-up', 'kimi-k2-dec-b64-down', 'kimi-k2-dec-b64-up', 'qwen3-235b-dec-b256-down', 'qwen3-235b-dec-b256-up', 'qwen3-235b-dec-b64-up']: return _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_factory(case)
    if _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_old is not None: return _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['MoEGroupedGemmFP8FwdOp'] = _integration700_transformer_dense_02_MoEGroupedGemmFP8FwdOp_dispatch

# integration700_transformer_dense verified final keys 01
from .integration700_transformer_dense_quant import quant as _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_factory
_integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_old = PROVIDERS.get('GemmINT8W8A8FwdOp')
def _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_dispatch(case):
    if case.id in ['llama3-8b-prefill-qkv-bfloat16']: return _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_factory(case)
    if _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_old is not None: return _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['GemmINT8W8A8FwdOp'] = _integration700_transformer_dense_final01_GemmINT8W8A8FwdOp_dispatch

# integration700_transformer_dense verified final keys 02
from .integration700_transformer_dense_quant import quant as _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_factory
_integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_old = PROVIDERS.get('GemmINT8W8A8FwdOp')
def _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_dispatch(case):
    if case.id in ['llama3-8b-prefill-down-bfloat16']: return _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_factory(case)
    if _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_old is not None: return _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['GemmINT8W8A8FwdOp'] = _integration700_transformer_dense_final02_GemmINT8W8A8FwdOp_dispatch
from .integration700_transformer_dense_fp8_bmm import fp8_bmm as _integration700_transformer_dense_final02_BmmFP8FwdOp_factory
_integration700_transformer_dense_final02_BmmFP8FwdOp_old = PROVIDERS.get('BmmFP8FwdOp')
def _integration700_transformer_dense_final02_BmmFP8FwdOp_dispatch(case):
    if case.id in ['square-1k-float8_e4m3fn-float16']: return _integration700_transformer_dense_final02_BmmFP8FwdOp_factory(case)
    if _integration700_transformer_dense_final02_BmmFP8FwdOp_old is not None: return _integration700_transformer_dense_final02_BmmFP8FwdOp_old(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase("No exact integration700 key")
PROVIDERS['BmmFP8FwdOp'] = _integration700_transformer_dense_final02_BmmFP8FwdOp_dispatch

# gmm_fp16_remaining11: frozen complete canonical pass, exact key only.
from .integration700_gmm_fp16_remaining11 import conv1d_verified as _gmm_fp16_remaining11_conv
_gmm_fp16_remaining11_conv_old = PROVIDERS.get('Conv1dFwdOp')
def _gmm_fp16_remaining11_conv_dispatch(case):
    if case.id == 'whisper-conv1-float16':
        return _gmm_fp16_remaining11_conv(case)
    return _gmm_fp16_remaining11_conv_old(case)
PROVIDERS['Conv1dFwdOp'] = _gmm_fp16_remaining11_conv_dispatch

# _gmm_fp16_remaining11_conv03_bias: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv03_bias as _gmm_fp16_remaining11_conv03_bias_factory
_gmm_fp16_remaining11_conv03_bias_old = PROVIDERS.get('Conv1dFwdOp')
def _gmm_fp16_remaining11_conv03_bias_dispatch(case):
    if case.id == 'whisper-conv1-bias-float16': return _gmm_fp16_remaining11_conv03_bias_factory(case)
    return _gmm_fp16_remaining11_conv03_bias_old(case)
PROVIDERS['Conv1dFwdOp'] = _gmm_fp16_remaining11_conv03_bias_dispatch

# _gmm_fp16_remaining11_hadamard01: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_hadamard01 as _gmm_fp16_remaining11_hadamard01_factory
_gmm_fp16_remaining11_hadamard01_old = PROVIDERS.get('HadamardTransformFwdOp')
def _gmm_fp16_remaining11_hadamard01_dispatch(case):
    if case.id == 'llama-8b-prefill-float16': return _gmm_fp16_remaining11_hadamard01_factory(case)
    return _gmm_fp16_remaining11_hadamard01_old(case)
PROVIDERS['HadamardTransformFwdOp'] = _gmm_fp16_remaining11_hadamard01_dispatch

# _gmm_fp16_remaining11_conv04_3d: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv04_3d as _gmm_fp16_remaining11_conv04_3d_factory
_gmm_fp16_remaining11_conv04_3d_old = PROVIDERS.get('Conv3dFwdOp')
def _gmm_fp16_remaining11_conv04_3d_dispatch(case):
    if case.id == 'unet-aspp-float16': return _gmm_fp16_remaining11_conv04_3d_factory(case)
    return _gmm_fp16_remaining11_conv04_3d_old(case)
PROVIDERS['Conv3dFwdOp'] = _gmm_fp16_remaining11_conv04_3d_dispatch

# _gmm_fp16_remaining11_conv05_video: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv05_video as _gmm_fp16_remaining11_conv05_video_factory
_gmm_fp16_remaining11_conv05_video_old = PROVIDERS.get('Conv3dFwdOp')
def _gmm_fp16_remaining11_conv05_video_dispatch(case):
    if case.id == 'video-down-bias-float16': return _gmm_fp16_remaining11_conv05_video_factory(case)
    return _gmm_fp16_remaining11_conv05_video_old(case)
PROVIDERS['Conv3dFwdOp'] = _gmm_fp16_remaining11_conv05_video_dispatch
