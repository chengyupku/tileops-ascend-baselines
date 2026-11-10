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
