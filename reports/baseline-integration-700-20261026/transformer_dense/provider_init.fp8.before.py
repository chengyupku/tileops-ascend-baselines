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
