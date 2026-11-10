from pathlib import Path
import sys,json,hashlib,fcntl,datetime,ast,shutil,difflib
P=Path(__file__).resolve().parent;W=Path('/opt/tileops-all-pairs/workspace');D=W/'TileOPs/benchmarks/ascend_all_pairs/tla'
group,adapter,batch=sys.argv[1:4];suffix=sys.argv[4] if len(sys.argv)>4 else 'v3';keys=json.loads((P/batch).read_text())
rows=[]
for op,cid in keys:
 f=P/'runs'/(op+'_'+cid+'_'+suffix)/'checker.json'
 if not f.exists():continue
 c=json.loads(f.read_text());g=json.loads((f.parent/'guard.json').read_text())
 if c.get('status')=='verified' and g['returncode']==0 and c['verification']['checked_outputs']==c['verification']['total_outputs']:rows.append((op,cid,f,c))
assert rows
F=P/('frozen_'+group);F.mkdir(exist_ok=True)
for src,dst in [(adapter,'adapter.py'),('run_case.py','run_case.py')]:
 if not (F/dst).exists():shutil.copy2(P/src,F/dst)
 assert (F/dst).read_bytes()==(P/src).read_bytes()
shutil.copytree(P/'private_tla',F/'private_tla',dirs_exist_ok=True)
keys=[[op,cid] for op,cid,f,c in rows];params={op+'/'+cid:json.loads((f.parent/'native.json').read_text())['params'] for op,cid,f,c in rows}
C=W/'.cache/all-pairs/remote_tla_compiler';R=W/'.cache/all-pairs/full_rollout_20261024/remote_ready/server1/group1/recovery_resume25/tla_supplemental_1858/runtime_env2/runtime/package'
pins={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [F/'adapter.py',F/'run_case.py',F/'private_tla/native.py',F/'private_tla/deployment.py']}
for op,cid,f,c in rows:
 for path,sha in json.loads((f.parent/'native.json').read_text())['hashes'].items():
  if '/ascend_baselines/' in path:pins[path]=sha
config=dict(workspace=str(W),vendor=str(W/'ascend_baselines/tilelang-ascend'),compiler=str(C),runtime=str(R),private=str(F),output_root=str(P/('formal_'+group)),pins=pins,keys=keys,params=params)
s=(P/'provider_template.txt').read_text();a=s.index('def config():');b=s.index('\ndef inspect_pins():')
s=s[:a]+f'def config():\n    return {config!r}\n\ndef matches(op, case):\n    return [op, case.id] in config()["keys"]\n\n'+s[b:]
s=s.replace("'._verified_where_private'",repr('._integration700_tla_simple_tilelang_'+group+'_private'))
s=s[:s.index('def factory(case):')]+'''def factory(op, case):
    if not matches(op,case): raise ValueError("Outside exact verified scope")
    if json.loads(json.dumps(case.params)) != config()['params'][op+'/'+case.id]: raise ValueError("Manifest changed")
    native, tilelang, adapter, out = _prepare()
    return adapter.factory(case, op, native, tilelang, out)
'''
name='integration700_tla_simple_tilelang_'+group;alias='_i700_tla_simple_'+group
ast.parse(s)
with open('/tmp/baseline700-provider-tla.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 init=D/'__init__.py';before=init.read_text();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
 (P/('provider_init_'+stamp+'.bak')).write_text(before)
 marker='# BEGIN '+name;endmarker='# END '+name
 block=f"""
{marker}
from . import {name} as {alias}
{alias}_previous = {{op: PROVIDERS.get(op) for op in {sorted(set(k[0] for k in keys))!r}}}
def {alias}_dispatch(op, case):
    if {alias}.matches(op, case): return {alias}.factory(op, case)
    return {alias}_previous[op](case)
for _op in {alias}_previous:
    PROVIDERS[_op] = partial({alias}_dispatch, _op)
{endmarker}
"""
 if marker in before:
  a=before.index(marker);b=before.index(endmarker,a)+len(endmarker);after=before[:a]+block.strip()+before[b:]
 else:after=before+block
 ast.parse(after)
 (D/(name+'.py')).write_text(s);(P/(name+'.py')).write_text(s);init.write_text(after)
 (P/('registration_'+stamp+'.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=str(init),tofile=str(init))))
(P/('REGISTERED_'+group+'.json')).write_text(json.dumps(dict(keys=keys,factory='benchmarks.ascend_all_pairs.tla.'+name+':factory',pins=pins,module_hash=hashlib.sha256(s.encode()).hexdigest()),indent=2))
print('registered',keys)
