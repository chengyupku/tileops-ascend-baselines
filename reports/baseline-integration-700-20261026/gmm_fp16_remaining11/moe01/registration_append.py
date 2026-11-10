
# _gmm_fp16_remaining11_moe01: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_moe01 as _gmm_fp16_remaining11_moe01_factory
_gmm_fp16_remaining11_moe01_old = PROVIDERS.get('MoEGroupedGemmFwdOp')
def _gmm_fp16_remaining11_moe01_dispatch(case):
    if case.id == 'ds-v3-decode-mlp-up-float16': return _gmm_fp16_remaining11_moe01_factory(case)
    return _gmm_fp16_remaining11_moe01_old(case)
PROVIDERS['MoEGroupedGemmFwdOp'] = _gmm_fp16_remaining11_moe01_dispatch
