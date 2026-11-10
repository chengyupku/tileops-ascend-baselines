import json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;m=json.loads((P/'MERGE_MAP.json').read_text());checks=[]
for base in {r['key'][0] for r in m}:
 prefix='benchmarks.ascend_all_pairs.'+base.replace('-','_');rows=[r for r in m if r['key'][0]==base];ops={r['key'][1] for r in rows}
 for r in rows:
  mod=r['factory'].rsplit('.',1)[0];obj=types.ModuleType(mod);obj.make=lambda c,mod=mod:mod;sys.modules[mod]=obj
 prior=lambda c:'prior';providers={op:prior for op in ops};providers['unrelated_sentinel']=prior;ns=dict(__package__=prefix,PROVIDERS=providers)
 exec((P/'merge_packet'/(base+'_registration.append.py')).read_text(),ns)
 for r in rows:
  _,op,cid=r['key'];case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  assert providers[op](case)==r['factory'].rsplit('.',1)[0];case.id='unaccepted-sentinel';assert providers[op](case)=='prior';checks.append(dict(key=r['key'],exact_route=True,prior_preserved=True))
 assert providers['unrelated_sentinel'] is prior
(P/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(dict(checks=checks,device_imports=False,shared_writes=False),indent=2));print('static routes checked',len(checks))
