
# nn_recovery01: exact keys with guarded full checker receipts.
_nn_recovery01_FusedAddRMSNormFwdOp_prior = PROVIDERS.get('FusedAddRMSNormFwdOp')
def _nn_recovery01_FusedAddRMSNormFwdOp(case, _prior=_nn_recovery01_FusedAddRMSNormFwdOp_prior):
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-13b-prefill-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-405b-decode-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-405b-prefill-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-70b-decode-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-70b-prefill-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-8b-prefill-bfloat16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-405b-prefill-float16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-70b-prefill-float16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if case._call.signature.name == 'FusedAddRMSNormFwdOp' and case.id == 'llama-8b-prefill-float16':
        from .integration700_nn_recovery01_3ab7708f1d import make
        return make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['FusedAddRMSNormFwdOp'] = _nn_recovery01_FusedAddRMSNormFwdOp
_nn_recovery01_GemmW4A16FwdOp_prior = PROVIDERS.get('GemmW4A16FwdOp')
def _nn_recovery01_GemmW4A16FwdOp(case, _prior=_nn_recovery01_GemmW4A16FwdOp_prior):
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'chunked-prefill-tile-float16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-b32-float16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-b8-float16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-hbm-streaming-float16':
        from .integration700_nn_recovery01_037f902f67 import make
        return make(case)
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-long-k-float16':
        from .integration700_nn_recovery01_037f902f67 import make
        return make(case)
    if case._call.signature.name == 'GemmW4A16FwdOp' and case.id == 'decode-non-pow2-float16':
        from .integration700_nn_recovery01_037f902f67 import make
        return make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmW4A16FwdOp'] = _nn_recovery01_GemmW4A16FwdOp
_nn_recovery01_FusedAddLayerNormFwdOp_prior = PROVIDERS.get('FusedAddLayerNormFwdOp')
def _nn_recovery01_FusedAddLayerNormFwdOp(case, _prior=_nn_recovery01_FusedAddLayerNormFwdOp_prior):
    if case._call.signature.name == 'FusedAddLayerNormFwdOp' and case.id == 'llama-70b-decode-bfloat16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'FusedAddLayerNormFwdOp' and case.id == 'llama-70b-prefill-bfloat16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'FusedAddLayerNormFwdOp' and case.id == 'llama-8b-decode-bfloat16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'FusedAddLayerNormFwdOp' and case.id == 'llama-8b-prefill-bfloat16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if case._call.signature.name == 'FusedAddLayerNormFwdOp' and case.id == 'llama-13b-prefill-float32':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['FusedAddLayerNormFwdOp'] = _nn_recovery01_FusedAddLayerNormFwdOp
_nn_recovery01_GemmFP8FwdOp_prior = PROVIDERS.get('GemmFP8FwdOp')
def _nn_recovery01_GemmFP8FwdOp(case, _prior=_nn_recovery01_GemmFP8FwdOp_prior):
    if case._call.signature.name == 'GemmFP8FwdOp' and case.id == 'ds-v3-m1-down-2d-float8_e4m3fn-bfloat16':
        from .integration700_nn_recovery01_68ce198132 import make
        return make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact key')
PROVIDERS['GemmFP8FwdOp'] = _nn_recovery01_GemmFP8FwdOp
