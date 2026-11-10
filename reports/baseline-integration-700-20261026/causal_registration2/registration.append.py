# Parent-only publication under ops_transformer publication lock.
from .integration700_causal_registration2 import wrap as _wrap_causal_registration2
PROVIDERS['CausalConv1dUpdateFwdOp'] = _wrap_causal_registration2(PROVIDERS.get('CausalConv1dUpdateFwdOp'))
