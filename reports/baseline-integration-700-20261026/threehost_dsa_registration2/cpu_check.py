import sys,builtins,types,json,dataclasses,copy,hashlib,datetime,ast
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;attempts=[];orig=builtins.__import__
def guarded(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):
  attempts.append(name);raise AssertionError('Forbidden device import '+name)
 return orig(name,*a,**kw)
builtins.__import__=guarded
def audit(event,args):
 if event=='ctypes.dlopen':raise AssertionError('Forbidden dlopen')
sys.addaudithook(audit)
import build_contracts as B
import registration as M
# Original public baseline is CPU-lazy; inspect actual provider dictionary copy.
F=O.parent/'threehost_attention_moe53/remote_evidence/frozen/TileOPs'
sys.path.insert(0,str(F))
from benchmarks.ascend_baselines import Candidate,UnsupportedCase
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
checks=[]
def reject(fn,exc=UnsupportedCase):
 try:fn()
 except exc as e:
  if exc is UnsupportedCase:assert type(e) is UnsupportedCase
  return
 raise AssertionError('Expected '+exc.__name__)
cs=B.cases();print('2 true DSA manifest calls instantiated',flush=True)
checks.append('True frozen manifest Call/Case, full params/optional inputs/out/tensor shapes and generated values digests; no template operator')
before=dict(PROVIDERS);private=dict(PROVIDERS);M.install(private)
for c in cs:
 assert M.matches(c)
 candidate=private[c._call.signature.name](c);assert isinstance(candidate,Candidate);candidate.cleanup()
assert PROVIDERS==before
checks.append('2 actual public Candidate objects; lazy install against copied provider table; shared table unchanged')
count=0
for c in cs:
 for n in c._call.params:
  bad=B.Case(c.id,dataclasses.replace(c._call,params=dict(c._call.params,**{n:'wrong'})),None)
  reject(lambda:private[c._call.signature.name](bad));count+=1
 for n,s in c._call.specs.items():
  if s is None:continue
  for delta in [{'shape':(1,)},{'dtype':'float64'},{'cpu':not s.cpu},{'values':[999]}]:
   specs=dict(c._call.specs);specs[n]=dataclasses.replace(s,**delta);bad=B.Case(c.id,dataclasses.replace(c._call,specs=specs),None)
   reject(lambda:private[c._call.signature.name](bad));count+=1
  specs=dict(c._call.specs);specs[n]=None;bad=B.Case(c.id,dataclasses.replace(c._call,specs=specs),None)
  reject(lambda:private[c._call.signature.name](bad));count+=1
 seen=[];unknown=types.SimpleNamespace(id='unknown',_call=c._call,params=c.params)
 assert M.wrap(lambda case:seen.append(case) or 'previous')(unknown)=='previous' and seen==[unknown]
checks.append(str(count)+' real-contract parameter/shape/dtype/cpu/values/optional absence mutations rejected with public UnsupportedCase; unknown keys preserve previous')
data=M.snapshot();load=M.private_load(data);mod=load(M.DSA);assert callable(mod.factory)
checks.append('Unchanged DSA original factory definition and pinned closure load without torch or native initialization')
# Metadata-only lifecycle fixtures; never call the original native factory.
saved={n:getattr(M,n) for n in ['admit','private_load','require_guard','environment','loaded_libraries','validate_inputs','matches','contract']}
c=cs[0];frozen_contract=M.contract(c)
try:
 guard=['g'];calls=[];counts={'factory':0,'run':0,'cleanup':0}
 M.matches=lambda c:True;M.contract=lambda c:frozen_contract
 M.require_guard=lambda:guard[0];M.environment=lambda:'env';M.loaded_libraries=lambda:None;M.validate_inputs=lambda *a:None
 def admit(runtime=True):calls.append(runtime);return data,guard[0],'env'
 M.admit=admit
 def factory(c):
  counts['factory']+=1
  def run(*a):counts['run']+=1;raise ValueError('CPU fixture failure before post-call synchronization')
  def clean():counts['cleanup']+=1
  return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=clean)
 M.private_load=lambda d:lambda p:types.SimpleNamespace(factory=factory)
 cand=M.make(c);assert counts['factory']==0;reject(lambda:cand.implementation.run(),ValueError);cand.cleanup();assert counts=={'factory':1,'run':1,'cleanup':1} and calls==[True]
 reject(lambda:cand.implementation.run(),RuntimeError)
finally:
 for n,v in saved.items():setattr(M,n,v)
checks.append('CPU failure lifecycle: lazy construction; one initial runtime hash admission; exactly-once cleanup; closed candidate rejects reuse')
# Wrong host admission must fail before any import even with true inputs metadata.
for c in cs:
 args=[types.SimpleNamespace(shape=c._call.specs[n].shape,dtype=c._call.specs[n].dtype,device='npu:0') for n in c._call.signature.inputs]
 candidate=M.make(c);reject(lambda:candidate.implementation.run(*args),RuntimeError);candidate.cleanup()
checks.append('2 true DSA invocation contracts reject local wrong host before native import')
# Final frozen metadata integrity and source tamper tests use private temp files only.
import tempfile,shutil
oldroot=M.DATA_ROOT
with tempfile.TemporaryDirectory(dir=O) as td:
 t=Path(td)
 for n in M.DATA_HASHES:shutil.copyfile(O/n,t/n)
 M.DATA_ROOT=t
 try:
  for n in M.DATA_HASHES:
   raw=(t/n).read_bytes();(t/n).write_bytes(raw+b' ');reject(M.metadata,RuntimeError);(t/n).write_bytes(raw)
 finally:M.DATA_ROOT=oldroot
with tempfile.TemporaryDirectory(dir=O) as td:
 p=Path(td)/'source';p.write_text('changed');reject(lambda:M.checked(p,'0'*64),RuntimeError)
checks.append('CONTRACTS/PINS/FROZEN metadata and source bytes tamper fail closed')
assert attempts==[]
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',registration_sha256=hashlib.sha256((O/'registration.py').read_bytes()).hexdigest(),checks=checks,device_import_attempts=attempts,numerical_runs=0,native_calls=0,accepted_added=0)
(O/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
