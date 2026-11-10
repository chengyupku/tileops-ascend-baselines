# Exact numerically verified SGL attention recovery keys; preserve all prior routes.
from pathlib import Path as _ar14_Path
_ar14_root = _ar14_Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_register14')
_ar14_keys = {'GQAPrefillPagedWithKVCacheFwdOp': ['qwen35-9b-rope-float16'], 'MHADecodePagedWithKVCacheFwdOp': ['serving-kv1k-float16', 'serving-kv512-float16'], 'MLAKVCacheWriteFwdOp': ['ds-v3-decode-bfloat16-bfloat16', 'kimi-k2-rope-bfloat16-bfloat16-bfloat16'], 'DSADecodeWithKVCacheFwdOp': ['mla-d512-h128-kv8k-bfloat16', 'mla-d576-h128-kv8k-bfloat16'], 'GQAVarlenFwdOp': ['gemma4-26b-a4b-ragged-bfloat16', 'gemma4-31b-uniform-float16'], 'MLADecodeWithKVCacheFwdOp': ['ds-v2-32k-float16', 'mla-b128-16384-bfloat16', 'mla-b128-16384-float16', 'mla-b128-32768-bfloat16', 'mla-b128-32768-float16']}
def _ar14_route(op, keys, previous):
    def route(case):
        if case.id not in keys:
            if previous is not None:
                return previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No registered SGL attention implementation for this key')
        import sys, importlib
        import benchmarks.ascend_baselines as base
        saved = base.LIBRARIES
        base.LIBRARIES = _ar14_root / 'libraries'
        sys.path.insert(0, str(_ar14_root))
        try:
            module = importlib.import_module('integration700_sgl_attention_register14')
            assert _ar14_Path(module.__file__).resolve().parent == _ar14_root
            return module.PROVIDERS[op](case)
        finally:
            base.LIBRARIES = saved
            sys.path.remove(str(_ar14_root))
    return route
for _ar14_op, _ar14_cases in _ar14_keys.items():
    PROVIDERS[_ar14_op] = _ar14_route(_ar14_op, frozenset(_ar14_cases), PROVIDERS.get(_ar14_op))
