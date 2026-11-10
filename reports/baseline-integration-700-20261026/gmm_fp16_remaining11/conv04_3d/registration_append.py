
# _gmm_fp16_remaining11_conv04_3d: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv04_3d as _gmm_fp16_remaining11_conv04_3d_factory
_gmm_fp16_remaining11_conv04_3d_old = PROVIDERS.get('Conv3dFwdOp')
def _gmm_fp16_remaining11_conv04_3d_dispatch(case):
    if case.id == 'unet-aspp-float16': return _gmm_fp16_remaining11_conv04_3d_factory(case)
    return _gmm_fp16_remaining11_conv04_3d_old(case)
PROVIDERS['Conv3dFwdOp'] = _gmm_fp16_remaining11_conv04_3d_dispatch
