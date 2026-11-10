# Parent publication only: frozen five original FlashAttn exactkeys.
from .integration700_attention_registration5 import wrap as _wrap_attention_registration5
for _attention_op in ('GQAPagedFwdOp', 'GQAVarlenFwdOp'):
    PROVIDERS[_attention_op] = _wrap_attention_registration5(PROVIDERS.get(_attention_op))
