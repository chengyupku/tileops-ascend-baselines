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
