
# _gmm_fp16_remaining11_hadamard01: complete canonical verification, exact key only.
from .integration700_gmm_fp16_remaining11 import verified_hadamard01 as _gmm_fp16_remaining11_hadamard01_factory
_gmm_fp16_remaining11_hadamard01_old = PROVIDERS.get('HadamardTransformFwdOp')
def _gmm_fp16_remaining11_hadamard01_dispatch(case):
    if case.id == 'llama-8b-prefill-float16': return _gmm_fp16_remaining11_hadamard01_factory(case)
    return _gmm_fp16_remaining11_hadamard01_old(case)
PROVIDERS['HadamardTransformFwdOp'] = _gmm_fp16_remaining11_hadamard01_dispatch
