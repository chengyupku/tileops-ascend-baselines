from pathlib import Path
import json,collections,difflib,sys
P=Path(__file__).resolve().parent;rows=json.loads((P/'RESULT.json').read_text())['rows'];group=collections.defaultdict(list)
exclude=set()
if '--remaining-only' in sys.argv:
 for n in ['DELIVERY_INTERIM.json','DELIVERY_DELTA03.json']:
  if (P/n).exists():exclude.update((r['operator'],r['case_id']) for r in json.loads((P/n).read_text())['rows'])
for r in rows:
 if r['status']=='verified_unregistered' and (r['operator'],r['case_id']) not in exclude:group[r['operator']].append(r['case_id'])
modules={'MLAKVCacheWriteFwdOp':'integration700_sgl_attention32','PagedKVCacheGatherFwdOp':'gather_adapter','DSADecodeWithKVCacheFwdOp':'dsa_adapter','GQAVarlenFwdOp':'attention_page16','GQAPrefillPagedWithKVCacheFwdOp':'prefill_adapter','MLADecodeWithKVCacheFwdOp':'attention_adapter','MHADecodePagedWithKVCacheFwdOp':'attention_adapter'}
text='''
# Exact numerically verified SGL attention recovery keys; retain other routes.
from pathlib import Path as _ar32_Path
_ar32_root = _ar32_Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/sgl_attention_recovery32')
'''+f'_ar32_keys = {dict(group)!r}\n_ar32_modules = {modules!r}\n'+'''
def _ar32_route(op, keys, previous):
    def route(case):
        if case.id not in keys:
            if previous is not None:
                return previous(case)
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('No registered SGL attention implementation for this key')
        import sys, importlib
        import benchmarks.ascend_baselines as base
        saved = base.LIBRARIES
        base.LIBRARIES = _ar32_root / 'libraries'
        sys.path.insert(0, str(_ar32_root))
        try:
            module = importlib.import_module(_ar32_modules[op])
            assert _ar32_Path(module.__file__).resolve().parent == _ar32_root
            return module.PROVIDERS[op](case)
        finally:
            base.LIBRARIES = saved
            sys.path.remove(str(_ar32_root))
    return route
for _ar32_op, _ar32_cases in _ar32_keys.items():
    PROVIDERS[_ar32_op] = _ar32_route(_ar32_op, frozenset(_ar32_cases), PROVIDERS.get(_ar32_op))
'''
if '--remaining-only' in sys.argv:text=text.replace('_ar32','_ar32_final07')
(P/'provider_registration.append.py').write_text(text)
before=(P/'provider_init.before_final.py').read_text()
(P/'REGISTRATION.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),(before+text).splitlines(True),fromfile='a/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py',tofile='b/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py')))
(P/'REGISTRATION_INSTRUCTIONS.txt').write_text('Parent review only. Under shared provider lock, reread current provider __init__.py and append exact provider_registration.append.py bytes, preserving all current registrations. REGISTRATION.patch is against provider_init.before.py and may require context refresh if another task registers first. Private adapters and original source symlink must remain at frozen paths. Canonical checker still needs the disclosed private diagnostic-only numerics override; no shared numerics modification proposed. Worker registered no keys.\n')
print('registration draft exact keys',sum(map(len,group.values())))
