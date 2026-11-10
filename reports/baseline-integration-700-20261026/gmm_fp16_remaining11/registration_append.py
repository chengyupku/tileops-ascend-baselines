
# gmm_fp16_remaining11: frozen complete canonical pass, exact key only.
from .integration700_gmm_fp16_remaining11 import conv1d_verified as _gmm_fp16_remaining11_conv
_gmm_fp16_remaining11_conv_old = PROVIDERS.get('Conv1dFwdOp')
def _gmm_fp16_remaining11_conv_dispatch(case):
    if case.id == 'whisper-conv1-float16':
        return _gmm_fp16_remaining11_conv(case)
    return _gmm_fp16_remaining11_conv_old(case)
PROVIDERS['Conv1dFwdOp'] = _gmm_fp16_remaining11_conv_dispatch
