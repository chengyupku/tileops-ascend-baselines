
# Exact keys from nn_recovery02; pending parent review and registration.
from .integration700_nn_recovery02_verified import make as _nn_recovery02_make, ACCEPTED_KEYS as _nn_recovery02_keys
_nn_recovery02_prior = {}
def _nn_recovery02_route(op, prior):
    def factory(case):
        if (case._call.signature.name, case.id) in _nn_recovery02_keys:
            return _nn_recovery02_make(case)
        if prior is not None:
            return prior(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No accepted exact key')
    return factory
for _nn_recovery02_op in sorted({key[0] for key in _nn_recovery02_keys}):
    _nn_recovery02_prior[_nn_recovery02_op] = PROVIDERS.get(_nn_recovery02_op)
    PROVIDERS[_nn_recovery02_op] = _nn_recovery02_route(_nn_recovery02_op, PROVIDERS.get(_nn_recovery02_op))
