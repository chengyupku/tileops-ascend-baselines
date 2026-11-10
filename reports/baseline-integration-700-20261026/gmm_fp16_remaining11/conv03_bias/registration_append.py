
# _gmm_fp16_remaining11_conv03_bias: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_conv03_bias as _gmm_fp16_remaining11_conv03_bias_factory
_gmm_fp16_remaining11_conv03_bias_old = PROVIDERS.get('Conv1dFwdOp')
def _gmm_fp16_remaining11_conv03_bias_dispatch(case):
    if case.id == 'whisper-conv1-bias-float16': return _gmm_fp16_remaining11_conv03_bias_factory(case)
    return _gmm_fp16_remaining11_conv03_bias_old(case)
PROVIDERS['Conv1dFwdOp'] = _gmm_fp16_remaining11_conv03_bias_dispatch
