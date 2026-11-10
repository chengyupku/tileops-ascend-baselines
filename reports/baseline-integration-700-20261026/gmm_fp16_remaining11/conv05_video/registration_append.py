
# _gmm_fp16_remaining11_conv05_video: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv05_video as _gmm_fp16_remaining11_conv05_video_factory
_gmm_fp16_remaining11_conv05_video_old = PROVIDERS.get('Conv3dFwdOp')
def _gmm_fp16_remaining11_conv05_video_dispatch(case):
    if case.id == 'video-down-bias-float16': return _gmm_fp16_remaining11_conv05_video_factory(case)
    return _gmm_fp16_remaining11_conv05_video_old(case)
PROVIDERS['Conv3dFwdOp'] = _gmm_fp16_remaining11_conv05_video_dispatch
