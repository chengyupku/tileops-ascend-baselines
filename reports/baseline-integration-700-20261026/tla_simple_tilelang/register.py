from pathlib import Path
import json,hashlib,fcntl,datetime,ast
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');D=W/'TileOPs/benchmarks/ascend_all_pairs/tla'
rows=[]
for f in sorted((P/'runs').glob('*/checker.json')):
 c=json.loads(f.read_text())
 if c.get('status')=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']:
  g=json.loads((f.parent/'guard.json').read_text())
  if g['returncode']==0: rows.append((f,c))
assert rows
F=P/'frozen_select_v1'
import shutil
F.mkdir(exist_ok=True)
for n in ['adapter.py','run_case.py']:
 if not (F/n).exists(): shutil.copy2(P/n,F/n)
shutil.copytree(P/'private_tla',F/'private_tla',dirs_exist_ok=True)
params={};keys=[]
for f,c in rows:
 n=json.loads((f.parent/'native.json').read_text())
 op,cid=f.parent.name.split('_',1);keys.append([op,cid]);params[op+'/'+cid]=n['params']
C=W/'.cache/all-pairs/remote_tla_compiler';R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
pins={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [F/'adapter.py',F/'run_case.py',F/'private_tla/native.py',F/'private_tla/deployment.py',W/'ascend_baselines/tilelang-ascend/testing/python/language/test_tilelang_ascend_language_select.py']}
config=dict(workspace=str(W),vendor=str(W/'ascend_baselines/tilelang-ascend'),compiler=str(C),runtime=str(R),private=str(F),output_root=str(P/'formal_runs'),pins=pins,keys=keys,params=params)
s=(P/'provider_template.txt').read_text()
start=s.index('def config():');end=s.index('\ndef inspect_pins():')
s=s[:start]+f'def config():\n    return {config!r}\n\ndef matches(op, case):\n    return [op, case.id] in config()["keys"]\n\n'+s[end:]
s=s.replace("'._verified_where_private'","'._integration700_tla_simple_tilelang_private'")
start=s.index('def factory(case):')
s=s[:start]+'''def factory(op, case):
    if not matches(op,case): raise ValueError("Outside exact verified scope")
    if json.loads(json.dumps(case.params)) != config()['params'][op+'/'+case.id]:
        raise ValueError("Manifest changed")
    native, tilelang, adapter, out = _prepare()
    return adapter.factory(case, op, native, tilelang, out)
'''
name='integration700_tla_simple_tilelang_select'
ast.parse(s);(D/(name+'.py')).write_text(s);(P/(name+'.py')).write_text(s)
with open('/tmp/baseline700-provider-tla.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 init=D/'__init__.py';before=init.read_text()
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
 (P/('provider_init_'+stamp+'.bak')).write_text(before)
 marker='# BEGIN integration700_tla_simple_tilelang_select'
 endmarker='# END integration700_tla_simple_tilelang_select'
 block=f"""
{marker}
from . import {name} as _i700_tla_simple_select
_i700_tla_simple_previous = {{op: PROVIDERS.get(op) for op in {sorted(set(k[0] for k in keys))!r}}}
def _i700_tla_simple_dispatch(op, case):
    if _i700_tla_simple_select.matches(op, case):
        return _i700_tla_simple_select.factory(op, case)
    return _i700_tla_simple_previous[op](case)
for _op in _i700_tla_simple_previous:
    PROVIDERS[_op] = partial(_i700_tla_simple_dispatch, _op)
{endmarker}
"""
 if marker in before:
  a=before.index(marker);b=before.index(endmarker,a)+len(endmarker)
  after=before[:a]+block.strip()+before[b:]
 else:after=before+block
 ast.parse(after);init.write_text(after)
 import difflib
 (P/('registration_'+stamp+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(init),tofile=str(init))))
(P/'REGISTERED.json').write_text(json.dumps(dict(keys=keys,factory='benchmarks.ascend_all_pairs.tla.'+name+':factory',pins=pins,module_hash=hashlib.sha256(s.encode()).hexdigest()),indent=2))
print('registered',keys)
