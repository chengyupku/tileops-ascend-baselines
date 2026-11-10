
# sgl_attention_register10: exact ten audited full canonical successes.
from .integration700_sgl_attention_register10 import build as _ar10_build, KEYS as _ar10_keys
def _ar10_wrap(previous):
    def route(case):
        if case.id in _ar10_keys:
            return _ar10_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _ar10_wrap(PROVIDERS.get('DSADecodeWithKVCacheFwdOp'))
