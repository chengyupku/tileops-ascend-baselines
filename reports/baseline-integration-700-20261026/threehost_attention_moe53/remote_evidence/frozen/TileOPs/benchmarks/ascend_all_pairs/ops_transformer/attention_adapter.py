"""Exact verified GQA keys, original function with private imports only."""
from .validated_runtime import MANIFEST, contract, fallback, preflight, load
RECIPE = MANIFEST['groups']['attention']
def gqa(case):
    if not contract(case, 'attention'):
        return fallback(case)
    preflight(case, 'attention')
    return load('ascend_transformer_graph').transformer_graph_gqa(case)
