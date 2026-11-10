import builtins,json,types,hashlib,sys
from pathlib import Path
O=Path(__file__).resolve().parent
source=(O/'integration700_gdn_chunk_registration16.py').read_bytes();ns={'__file__':str(O/'integration700_gdn_chunk_registration16.py')};exec(compile(source,ns['__file__'],'exec'),ns)
checks=[];events=[];Case=types.SimpleNamespace
original=builtins.__import__
def blocked(name,*a,**kw):
 if name in fake:return fake[name]
 if name.split('.')[0] in ('torch','torch_npu','acl'):raise AssertionError('real device import forbidden')
 return original(name,*a,**kw)
class Transport:
 def __init__(self,*a):events.append(('api',a[0]))
 def cleanup(self):events.append(('cleanup',))
 def call(self,*a):raise AssertionError('numerics forbidden')
class Candidate:
 def __init__(self,*a,**kw):self.args=a;self.kw=kw
fake={'torch':types.ModuleType('torch'),'benchmarks.api':Case(Implementation=lambda f:f),'benchmarks.ascend_baselines':Case(Candidate=Candidate)}
def reject(f,n):
 try:f()
 except (ValueError,RuntimeError,KeyError,FileNotFoundError):checks.append(n);return
 raise AssertionError(n)
def guard():events.append(('guard',))
real=ns['_execute']
def execute(name,file,raw,transport=None):
 if name=='_frozen_gdn_transport':return Case(require_guard=guard,DirectACL=Transport)
 return real(name,file,raw,transport)
builtins.__import__=blocked
try:
 snap=ns['_snapshot']();checks.append('original26 file pins')
 for cid,p in ns['CASE_PARAMS'].items():
  case=Case(id=cid,params=json.loads(p))
  for k in case.params:
   altered=dict(case.params);altered[k]='changed';reject(lambda:ns['build'](Case(id=cid,params=altered)),cid+' reject '+k)
  for altered in (dict(case.params,extra=True),{k:v for k,v in case.params.items() if k!='scale'},dict(case.params,state_v_first=0)):
   reject(lambda:ns['build'](Case(id=cid,params=altered)),cid+' shape/type rejection')
  reject(lambda:ns['build'](case),cid+' original guard rejects CPU before torch')
 class Corrupt:
  def read_bytes(self):return b'changed'
  def __str__(self):return 'synthetic CPU corruption'
 for n,p in ns['FILE_PINS'].items():reject(lambda:ns['_checked'](Corrupt(),p),'pin rejects '+n)
 ns['_execute']=execute;ns['_environment']=lambda:None;ns['RUNTIME_PINS']={}
 for cid,p in ns['CASE_PARAMS'].items():
  result=ns['wrap'](None)(Case(id=cid,params=json.loads(p)))
  assert result.args[1]=='ops-transformer::aclnnChunkGatedDeltaRule';assert result.kw['timing_mode']=='host_roundtrip';result.kw['cleanup']();checks.append(cid+' exact original binding API construction cleanup')
 assert events.count(('api','ChunkGatedDeltaRule'))==16 and events.count(('cleanup',))==16 and events.count(('guard',))==32
 assert ns['wrap'](lambda c:'fallback')(Case(id='other'))=='fallback';reject(lambda:ns['wrap'](None)(Case(id='other')),'missing fallback')
 assert ns['_snapshot']()==snap;checks.append('config snapshots unchanged')
 # Snapshot stays fixed even when underlying file would subsequently change.
 fp=ns['_FrozenPath'](Path('/synthetic'),{'/synthetic':b'original'});assert fp.read_text()=='original';checks.append('FrozenPath reads retained bytes')
finally:builtins.__import__=original
(O/'CPU_PROOF.json').write_text(json.dumps({'passed':len(checks),'checks':checks,'events':events,'module_sha256':hashlib.sha256(source).hexdigest(),'native_imports':0,'npu_runs':0,'numerical_reruns':0,'boundary':'Test namespace substitutions only; original binding executed, numerical callable never invoked.'},indent=2)+'\n')
print('passed',len(checks))
