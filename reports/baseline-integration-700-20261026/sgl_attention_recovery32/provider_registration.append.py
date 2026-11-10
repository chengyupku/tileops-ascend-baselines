
# Exact numerically verified SGL attention recovery keys; retain other routes.
from pathlib import Path as _ar32_final07_Path
_ar32_final07_root = _ar32_final07_Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_recovery32')
_ar32_final07_keys = {'DSADecodeWithKVCacheFwdOp': ['mla-d512-h64-kv32k-bfloat16', 'mla-d512-h64-kv48k-bfloat16', 'mla-d512-h64-kv64k-bfloat16', 'mla-d576-h128-kv128k-bfloat16', 'mla-d576-h128-kv32k-bfloat16', 'mla-d576-h128-kv64k-bfloat16', 'mla-d576-h128-kv96k-bfloat16']}
_ar32_final07_modules = {'MLAKVCacheWriteFwdOp': 'integration700_sgl_attention32', 'PagedKVCacheGatherFwdOp': 'gather_adapter', 'DSADecodeWithKVCacheFwdOp': 'dsa_adapter', 'GQAVarlenFwdOp': 'attention_page16', 'GQAPrefillPagedWithKVCacheFwdOp': 'prefill_adapter', 'MLADecodeWithKVCacheFwdOp': 'attention_adapter', 'MHADecodePagedWithKVCacheFwdOp': 'attention_adapter'}

def _ar32_final07_route(op, keys, previous):
    def route(case):
        if case.id not in keys:
            if previous is not None:
                return previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No registered SGL attention implementation for this key')
        import sys, importlib
        import benchmarks.ascend_baselines as base
        saved = base.LIBRARIES
        base.LIBRARIES = _ar32_final07_root / 'libraries'
        sys.path.insert(0, str(_ar32_final07_root))
        try:
            module = importlib.import_module(_ar32_final07_modules[op])
            assert _ar32_final07_Path(module.__file__).resolve().parent == _ar32_final07_root
            return module.PROVIDERS[op](case)
        finally:
            base.LIBRARIES = saved
            sys.path.remove(str(_ar32_final07_root))
    return route
for _ar32_final07_op, _ar32_final07_cases in _ar32_final07_keys.items():
    PROVIDERS[_ar32_final07_op] = _ar32_final07_route(_ar32_final07_op, frozenset(_ar32_final07_cases), PROVIDERS.get(_ar32_final07_op))
