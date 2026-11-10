"""Lazy tilelang-ascend providers. Importing this entry performs no device imports."""
from functools import partial
from typing import TYPE_CHECKING, Callable

from .plans import IMPLEMENTED

if TYPE_CHECKING:
    from benchmarks.api import Case
    from benchmarks.ascend_baselines import Candidate


def _make(op,case):
    from .verified_where import matches, factory as verified_factory
    if matches(op, case):
        return verified_factory(case)
    from .native import prepare_deployment
    prepare_deployment()
    from .adapters import factory
    return factory(op,case)


PROVIDERS: dict[str, Callable[['Case'], 'Candidate']] = {
    op: partial(_make,op) for op in sorted(IMPLEMENTED)
}

# BEGIN integration700_tla_simple_tilelang_select
from . import integration700_tla_simple_tilelang_select as _i700_tla_simple_select
_i700_tla_simple_previous = {op: PROVIDERS.get(op) for op in ['WhereFwdOp']}
def _i700_tla_simple_dispatch(op, case):
    if _i700_tla_simple_select.matches(op, case):
        return _i700_tla_simple_select.factory(op, case)
    return _i700_tla_simple_previous[op](case)
for _op in _i700_tla_simple_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_dispatch, _op)
# END integration700_tla_simple_tilelang_select

# BEGIN integration700_tla_simple_tilelang_select3
from . import integration700_tla_simple_tilelang_select3 as _i700_tla_simple_select3
_i700_tla_simple_select3_previous = {op: PROVIDERS.get(op) for op in ['MaskedFillScalarFwdOp', 'MaskedFillTensorFwdOp']}
def _i700_tla_simple_select3_dispatch(op, case):
    if _i700_tla_simple_select3.matches(op, case): return _i700_tla_simple_select3.factory(op, case)
    return _i700_tla_simple_select3_previous[op](case)
for _op in _i700_tla_simple_select3_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_select3_dispatch, _op)
# END integration700_tla_simple_tilelang_select3

# BEGIN integration700_tla_simple_tilelang_element5
from . import integration700_tla_simple_tilelang_element5 as _i700_tla_simple_element5
_i700_tla_simple_element5_previous = {op: PROVIDERS.get(op) for op in ['AbsFwdOp', 'NegFwdOp']}
def _i700_tla_simple_element5_dispatch(op, case):
    if _i700_tla_simple_element5.matches(op, case): return _i700_tla_simple_element5.factory(op, case)
    return _i700_tla_simple_element5_previous[op](case)
for _op in _i700_tla_simple_element5_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_element5_dispatch, _op)
# END integration700_tla_simple_tilelang_element5

# BEGIN integration700_tla_simple_tilelang_element6
from . import integration700_tla_simple_tilelang_element6 as _i700_tla_simple_element6
_i700_tla_simple_element6_previous = {op: PROVIDERS.get(op) for op in ['BitwiseNotFwdOp', 'ClampScalarFwdOp', 'ExpFwdOp', 'LogFwdOp', 'LogicalNotFwdOp', 'ReciprocalFwdOp', 'RsqrtFwdOp']}
def _i700_tla_simple_element6_dispatch(op, case):
    if _i700_tla_simple_element6.matches(op, case): return _i700_tla_simple_element6.factory(op, case)
    return _i700_tla_simple_element6_previous[op](case)
for _op in _i700_tla_simple_element6_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_element6_dispatch, _op)
# END integration700_tla_simple_tilelang_element6

# BEGIN integration700_tla_simple_tilelang_sqrt7
from . import integration700_tla_simple_tilelang_sqrt7 as _i700_tla_simple_sqrt7
_i700_tla_simple_sqrt7_previous = {op: PROVIDERS.get(op) for op in ['SqrtFwdOp']}
def _i700_tla_simple_sqrt7_dispatch(op, case):
    if _i700_tla_simple_sqrt7.matches(op, case): return _i700_tla_simple_sqrt7.factory(op, case)
    return _i700_tla_simple_sqrt7_previous[op](case)
for _op in _i700_tla_simple_sqrt7_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_sqrt7_dispatch, _op)
# END integration700_tla_simple_tilelang_sqrt7

# BEGIN integration700_tla_simple_tilelang_gemm7
from . import integration700_tla_simple_tilelang_gemm7 as _i700_tla_simple_gemm7
_i700_tla_simple_gemm7_previous = {op: PROVIDERS.get(op) for op in ['BmmFwdOp', 'GemmFwdOp']}
def _i700_tla_simple_gemm7_dispatch(op, case):
    if _i700_tla_simple_gemm7.matches(op, case): return _i700_tla_simple_gemm7.factory(op, case)
    return _i700_tla_simple_gemm7_previous[op](case)
for _op in _i700_tla_simple_gemm7_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_gemm7_dispatch, _op)
# END integration700_tla_simple_tilelang_gemm7

# BEGIN integration700_tla_simple_tilelang_moe7
from . import integration700_tla_simple_tilelang_moe7 as _i700_tla_simple_moe7
_i700_tla_simple_moe7_previous = {op: PROVIDERS.get(op) for op in ['MoEGroupedGemmFwdOp']}
def _i700_tla_simple_moe7_dispatch(op, case):
    if _i700_tla_simple_moe7.matches(op, case): return _i700_tla_simple_moe7.factory(op, case)
    return _i700_tla_simple_moe7_previous[op](case)
for _op in _i700_tla_simple_moe7_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_moe7_dispatch, _op)
# END integration700_tla_simple_tilelang_moe7

# BEGIN integration700_tla_simple_tilelang_gemm_rest7
from . import integration700_tla_simple_tilelang_gemm_rest7 as _i700_tla_simple_gemm_rest7
_i700_tla_simple_gemm_rest7_previous = {op: PROVIDERS.get(op) for op in ['BmmFwdOp', 'GemmFwdOp']}
def _i700_tla_simple_gemm_rest7_dispatch(op, case):
    if _i700_tla_simple_gemm_rest7.matches(op, case): return _i700_tla_simple_gemm_rest7.factory(op, case)
    return _i700_tla_simple_gemm_rest7_previous[op](case)
for _op in _i700_tla_simple_gemm_rest7_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_gemm_rest7_dispatch, _op)
# END integration700_tla_simple_tilelang_gemm_rest7
