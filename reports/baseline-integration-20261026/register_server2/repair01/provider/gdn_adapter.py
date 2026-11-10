"""Exact verified GDN keys, original build and both original outputs."""
from .validated_runtime import contract, fallback, preflight, load
def gdn(case):
    if not contract(case, 'gdn'):
        return fallback(case)
    preflight(case, 'gdn')
    return load('binding').build(case)
