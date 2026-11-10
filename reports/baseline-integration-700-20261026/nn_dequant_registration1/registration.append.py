
# baseline700: one exact original Mul dequant receipt; preserve previous factory.
from . import integration700_nn_dequant_registration1 as _nn_dequant_registration1
_nn_dequant_registration1_previous = PROVIDERS.get('INT8DequantPerChannelFwdOp')
def _nn_dequant_registration1_route(case, _previous=_nn_dequant_registration1_previous):
    if (case._call.signature.name, case.id) == _nn_dequant_registration1.KEY[1:]:
        # Same key with wrong params must fail closed, not fall through to old route.
        return _nn_dequant_registration1.make(case, baseline='ops-math')
    if _previous is not None:
        return _previous(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No accepted exact dequant route')
PROVIDERS['INT8DequantPerChannelFwdOp'] = _nn_dequant_registration1_route
