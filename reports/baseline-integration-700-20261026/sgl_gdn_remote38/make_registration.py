from pathlib import Path
import json,difflib,hashlib
P=Path(__file__).resolve().parent
rows=json.loads((P/'RESULT.json').read_text())['rows'];keys={}
for r in rows:
 if r['status']=='verified_unregistered':keys.setdefault(r['operator'],[]).append(r['case_id'])
module='integration700_sgl_gdn_remote38'; target='TileOPs/benchmarks/ascend_all_pairs/sgl/'+module+'.py'
s=(P/(module+'.py')).read_text();patch=''.join(difflib.unified_diff([],s.splitlines(True),fromfile='/dev/null',tofile='b/'+target))
base=(P/'provider_init.before.py').read_text()
addition='\n# Exact verified keys from sgl_gdn_remote38; server2 source binding.\nfrom .'+module+' import build as _gdn38_build\n_gdn38_keys = '+repr(keys)+'''\ndef _gdn38_route(previous, keys):
    def route(case):
        if case.id in keys:
            return _gdn38_build(case)
        if previous is not None:
            return previous(case)
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No SGL provider for this unregistered exact key')
    return route
for _gdn38_op, _gdn38_cases in _gdn38_keys.items():
    PROVIDERS[_gdn38_op] = _gdn38_route(PROVIDERS.get(_gdn38_op), frozenset(_gdn38_cases))
'''
patch+=''.join(difflib.unified_diff(base.splitlines(True),(base+addition).splitlines(True),fromfile='a/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py',tofile='b/TileOPs/benchmarks/ascend_all_pairs/sgl/__init__.py'))
(P/'REGISTRATION.patch').write_text(patch);(P/'RECOMMENDED_ACCEPT.json').write_text(json.dumps({'host':'server2','registered':False,'keys':keys,'count':sum(map(len,keys.values())),'adapter_sha256':hashlib.sha256(s.encode()).hexdigest(),'registration_patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'note':'Parent review only; no shared provider mutation. Absolute source binding is server2 original source.'},indent=2))
