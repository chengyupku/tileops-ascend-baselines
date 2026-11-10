"""Isolated TileLang providers. Importing this module does not import device runtimes."""
from .adapters import bmm

PROVIDERS = {"BmmFwdOp": bmm}

# integration700_tilelang_register5: exactly five previously checked original TileLang FP8 cases.
from .integration700_tilelang_register5 import factory as _register5_factory, KEYS as _register5_keys
_register5_previous = {op: PROVIDERS.get(op) for op, _ in _register5_keys}
def _register5_dispatch(case):
    op = case._call.signature.name
    if (op, case.id) in _register5_keys:
        return _register5_factory(case)
    previous = _register5_previous[op]
    if previous is None:
        raise NotImplementedError((op, case.id))
    return previous(case)
PROVIDERS.update({op: _register5_dispatch for op, _ in _register5_keys})
