import builtins, json, types, hashlib, sys
from pathlib import Path
O=Path(__file__).resolve().parent
source=(O/'integration700_gdn_recurrent_registration1.py').read_bytes()
ns={'__name__':'cpu_test_only','__file__':str(O/'integration700_gdn_recurrent_registration1.py')}
exec(compile(source,ns['__file__'],'exec'),ns)
Case=types.SimpleNamespace
case=Case(id=ns['CASE_ID'],params=json.loads(ns['PARAMS_JSON']))
checks=[]
def ok(name): checks.append(name)
def rejects(fn, name):
 try:fn()
 except (ValueError,RuntimeError,KeyError,FileNotFoundError):ok(name);return
 raise AssertionError(name)
assert ns['wrap'](lambda c:'fallback')(Case(id='other'))=='fallback';ok('fallback')
rejects(lambda:ns['wrap'](None)(Case(id='other')),'unknown without fallback')
for k in case.params:
 p=dict(case.params);p[k]='altered'
 rejects(lambda:ns['build'](Case(id=case.id,params=p)),'reject parameter '+k)
p=dict(case.params,extra=True);rejects(lambda:ns['build'](Case(id=case.id,params=p)),'reject extra parameter')
p=dict(case.params);p.pop('scale');rejects(lambda:ns['build'](Case(id=case.id,params=p)),'reject missing parameter')
p=dict(case.params);p['state_v_first']=0;rejects(lambda:ns['build'](Case(id=case.id,params=p)),'reject bool-int equivalence')
snapshot=ns['_snapshot']();ok('all original config hashes match')
class Corrupt:
 def read_bytes(self):return b'corrupt'
 def __str__(self):return 'CPU synthetic corrupted bytes'
for name,pin in ns['FILE_PINS'].items(): rejects(lambda:ns['_checked'](Corrupt(),pin),'reject damaged '+name)
# Deny native imports; supply explicit CPU modules solely inside this test process.
original_import=builtins.__import__;real_execute=ns['_execute'];events=[]
class Transport:
 def __init__(self,*args):self.args=args;events.append(('api',args[0]))
 def cleanup(self):events.append(('cleanup',))
 def call(self,*args):raise AssertionError('numerical execution forbidden')
def guard():events.append(('guard',))
class Candidate:
 def __init__(self,*args,**kw):self.args=args;self.kw=kw
fake={'torch':types.ModuleType('torch'),'benchmarks.api':types.SimpleNamespace(Implementation=lambda f:('CPU_ONLY_UNEXECUTABLE',f)),'benchmarks.ascend_baselines':types.SimpleNamespace(Candidate=Candidate)}
def controlled(name,*args,**kwargs):
 if name in fake:return fake[name]
 if name.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError('native import forbidden')
 return original_import(name,*args,**kwargs)
def execute(name,filename,raw,transport=None):
 if name=='_frozen_gdn_transport':return types.SimpleNamespace(require_guard=guard,DirectACL=Transport)
 return real_execute(name,filename,raw,transport)
# No production bypass flag exists. Test namespace boundaries alone are substituted.
builtins.__import__=controlled
try:
 rejects(lambda:ns['build'](case),'production guard denies CPU process before torch import')
 ns['_execute']=execute;ns['_environment']=lambda:events.append(('environment_boundary_stub',))
 ns['RUNTIME_PINS']={}
 result=ns['wrap'](None)(case)
 assert result.args[1]=='ops-transformer::aclnnRecurrentKda'
 assert result.kw['timing_mode']=='host_roundtrip';result.kw['cleanup']()
 assert events.count(('guard',))==2 and ('api','RecurrentKda') in events and ('cleanup',) in events
 ok('future formal route exact original binding bytes and original API/cleanup metadata')
 assert ns['_snapshot']()==snapshot;ok('original configuration unchanged')
finally:builtins.__import__=original_import
(O/'CPU_PROOF.json').write_text(json.dumps({'passed':len(checks),'checks':checks,'events':events,'module_sha256':hashlib.sha256(source).hexdigest(),'native_imports':0,'npu_runs':0,'numerical_reruns':0,'runtime_boundary':'Explicit test namespace substitutions only; no numerical callable invoked; no production stub switch.'},indent=2)+'\n')
print('CPU checks passed:',len(checks))
