
# sgl_dsa_source_reconcile1: canonical full-output pass, historical source race reconciled.
from .integration700_sgl_dsa_source_reconcile1 import build as _dsa_sr1_build, KEYS as _dsa_sr1_keys
def _dsa_sr1_wrap(previous):
    def route(case):
        if case.id in _dsa_sr1_keys:
            return _dsa_sr1_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
PROVIDERS['DSADecodeWithKVCacheFwdOp'] = _dsa_sr1_wrap(PROVIDERS.get('DSADecodeWithKVCacheFwdOp'))
