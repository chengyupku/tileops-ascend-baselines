import sys,builtins,json,dataclasses,types,datetime
from pathlib import Path
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
orig=builtins.__import__;attempts=[]
def protect(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','tilelang','tvm','acl'):
  attempts.append(name);raise AssertionError('Forbidden CPU import '+name)
 return orig(name,*a,**kw)
builtins.__import__=protect
def audit(event,args):
 if event=='ctypes.dlopen':raise AssertionError('Forbidden dlopen')
sys.addaudithook(audit)
import build_contracts as B
import registration as M
from benchmarks.ascend_baselines import Candidate,UnsupportedCase
from benchmarks.ascend_all_pairs.tla import PROVIDERS
before=dict(PROVIDERS);private=dict(PROVIDERS);M.install(private);count=0
def rejects(fn,exc):
 try:fn()
 except exc:return
 raise AssertionError('Expected '+exc.__name__)
for c in B.cases():
 candidate=private[c._call.signature.name](c)
 assert isinstance(candidate,Candidate)
 assert M.matches(c)
 ev=next(r for r in json.loads((P/'EVIDENCE.json').read_text()) if r['key'][1:]==[c._call.signature.name,c.id])
 # Receipts only on originating local archive; remote route check needs no archive.
 if Path(ev['checker']).exists():
  rec=json.loads(Path(ev['checker']).read_text())
  assert rec['case_snapshot']['parameters']==M.plain(c.params)
  assert rec['status']=='verified' and rec['inputs_preserved']
  assert rec['verification']['checked_outputs']==rec['verification']['total_outputs']
 bad=B.Case(c.id,dataclasses.replace(c._call,params=dict(c._call.params,out=True)),None)
 rejects(lambda:private[c._call.signature.name](bad),UnsupportedCase)
 for n,s in c._call.specs.items():
  if s is None:continue
  specs=dict(c._call.specs);specs[n]=dataclasses.replace(s,shape=(1,))
  bad=B.Case(c.id,dataclasses.replace(c._call,specs=specs),None)
  rejects(lambda:private[c._call.signature.name](bad),UnsupportedCase);count+=1
 args=[types.SimpleNamespace(shape=c._call.specs[n].shape,dtype=c._call.specs[n].dtype,device='npu:0') for n in c._call.signature.inputs]
 rejects(lambda:candidate.implementation.run(*args),RuntimeError)
 candidate.cleanup()
 unknown=types.SimpleNamespace(id='unknown',_call=c._call)
 assert M.wrap(lambda c:'fallback')(unknown)=='fallback'
 rejects(lambda:M.wrap(None)(unknown),UnsupportedCase)
assert before==PROVIDERS and not attempts
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,exact_keys=23,shape_mutations=count,device_imports=attempts,shared_provider_writes=0,npu_runs=0,accepted_added=0)
print(json.dumps(r));(P/'CPU_CHECK.json').write_text(json.dumps(r,indent=2)+'\n')
