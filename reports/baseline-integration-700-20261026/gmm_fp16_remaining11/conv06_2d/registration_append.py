
# _gmm_fp16_remaining11_conv06_2d: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv06_2d as _gmm_fp16_remaining11_conv06_2d_factory
_gmm_fp16_remaining11_conv06_2d_old = PROVIDERS.get('Conv2dFwdOp')
def _gmm_fp16_remaining11_conv06_2d_dispatch(case):
    if case.id == 'highres-float16': return _gmm_fp16_remaining11_conv06_2d_factory(case)
    return _gmm_fp16_remaining11_conv06_2d_old(case)
PROVIDERS['Conv2dFwdOp'] = _gmm_fp16_remaining11_conv06_2d_dispatch
