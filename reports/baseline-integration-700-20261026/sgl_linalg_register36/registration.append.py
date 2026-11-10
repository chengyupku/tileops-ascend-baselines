
# SGL linalg36: verified exact keys only; preserve every preexisting route.
from .integration700_sgl_linalg_register36 import build as _integration700_sgl_linalg_register36_build
_integration700_sgl_linalg_register36_keys = {'BmmFwdOp': ['mha-decode-pv-bfloat16', 'mha-decode-pv-float16', 'mha-decode-qk-bfloat16', 'mha-decode-qk-float16', 'moe-prefill-bfloat16', 'square-128-bfloat16', 'square-128-float16', 'square-1k-bfloat16', 'square-1k-float16', 'square-512-bfloat16', 'square-512-float16'], 'GemmFwdOp': ['ds-v3-decode-down-bfloat16', 'ds-v3-decode-qkv-a-bfloat16', 'ds-v3-m1-qkv-a-bfloat16', 'ds-v3-m16-mlp-up-bfloat16', 'ds-v3-m2-qkv-a-bfloat16', 'ds-v3-m32-mlp-up-bfloat16', 'ds-v3-m64-down-bfloat16', 'ds-v3-m96-qkv-a-bfloat16', 'ds-v3-prefill-down-bfloat16', 'ds-v3-prefill-mlp-up-bfloat16', 'ds-v3-prefill-mlp-up-float16', 'ds-v3-prefill-o-proj-bfloat16', 'ds-v3-prefill-q-b-bfloat16', 'ds-v3-prefill-qkv-a-bfloat16', 'qwen3-8b-down-prefill-bfloat16', 'qwen3-8b-gate-up-b32-bfloat16', 'qwen3-8b-gate-up-decode-bfloat16', 'square-1k-bfloat16', 'vit-l16-in21k-head-bfloat16'], 'GroupedGemmFwdOp': ['moe-dgrad-float16', 'moe-fwd-bfloat16', 'moe-fwd-float16', 'qwen3-30b-ep8-empty-bfloat16', 'qwen3-30b-ep8-ragged-bfloat16', 'qwen3-30b-ep8-routed1k-bfloat16']}
def _integration700_sgl_linalg_register36_route(previous, exact_keys):
    def route(case):
        if case.id in exact_keys:
            return _integration700_sgl_linalg_register36_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
for _integration700_sgl_linalg_register36_op, _integration700_sgl_linalg_register36_cases in _integration700_sgl_linalg_register36_keys.items():
    PROVIDERS[_integration700_sgl_linalg_register36_op] = _integration700_sgl_linalg_register36_route(
        PROVIDERS.get(_integration700_sgl_linalg_register36_op), frozenset(_integration700_sgl_linalg_register36_cases))
