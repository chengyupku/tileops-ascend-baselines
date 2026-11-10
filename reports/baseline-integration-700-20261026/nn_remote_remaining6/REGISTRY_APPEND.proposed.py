# Proposed append to ops_nn/__init__.py, under /tmp/baseline700-provider-ops_nn.lock.
# Not applied by this worker: local shared-provider writes are outside its allowed paths.
from . import integration700_nn_remote_remaining6 as _nn_remote_remaining6
for _nn_remote_remaining6_op in ('FusedAddRMSNormFwdOp', 'BatchNormFwdOp'):
    def _nn_remote_remaining6_route(case, _prior=PROVIDERS.get(_nn_remote_remaining6_op)):
        if (case._call.signature.name, case.id) in _nn_remote_remaining6._KEYS:
            return _nn_remote_remaining6.make(case)
        if _prior is not None:
            return _prior(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No registered exact key')
    PROVIDERS[_nn_remote_remaining6_op] = _nn_remote_remaining6_route
