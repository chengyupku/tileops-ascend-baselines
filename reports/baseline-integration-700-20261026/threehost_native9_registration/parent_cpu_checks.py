import sys,builtins,types,ctypes,json,dataclasses,copy,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
O=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/threehost_native9_registration');sys.path.insert(0,str(O))
blocked=[];dl=[];original=builtins.__import__
def protected(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):
  blocked.append(name);raise AssertionError('Device import '+name)
 return original(name,*args,**kwargs)
builtins.__import__=protected
def audit(event,args):
 if event=='ctypes.dlopen':dl.append(str(args));raise AssertionError('dlopen forbidden')
sys.addaudithook(audit)
import build_contracts as B
import registration_parent as M
from benchmarks.ascend_baselines import UnsupportedCase,Candidate
cs=B.cases();checks=[]
def ok(message):checks.append(message)
def rejects(fn,exc=UnsupportedCase):
 try:fn()
 except exc as e:
  if exc is UnsupportedCase:assert type(e) is UnsupportedCase
  return
 raise AssertionError('Expected '+exc.__name__)
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
# Published provider import and registration, only a private dictionary is mutated.
before=dict(PROVIDERS);published=dict(PROVIDERS);M.install(published)
for c in cs:
 cand=published[c._call.signature.name](c);assert isinstance(cand,Candidate) and callable(cand.implementation.run);cand.cleanup()
assert PROVIDERS==before
ok('9 true manifest Call + original passive Case through actual provider dictionary copy and install; Candidate lazy; shared dictionary unchanged')
mutations=0
for c in cs:
 def reject_call(call):
  global mutations
  bad=B.Case(c.id,call,None);rejects(lambda:published[c._call.signature.name](bad));mutations+=1
 for name in c._call.params:
  reject_call(dataclasses.replace(c._call,params=dict(c._call.params,**{name:'wrong'})))
 reject_call(dataclasses.replace(c._call,params=dict(c._call.params,bias=True)))
 for name in c._call.indices:
  reject_call(dataclasses.replace(c._call,indices=dict(c._call.indices,**{name:'wrong'})))
 for name,spec in c._call.specs.items():
  if spec is None:continue
  for fields in ({'shape':(1,)},{'dtype':'float64'},{'cpu':not spec.cpu},{'values':[999]}):
   specs=dict(c._call.specs);specs[name]=dataclasses.replace(spec,**fields);reject_call(dataclasses.replace(c._call,specs=specs))
  specs=dict(c._call.specs);specs[name]=None;reject_call(dataclasses.replace(c._call,specs=specs))
 bad=types.SimpleNamespace(id=c.id,_call=c._call,params=dict(c.params,layout='wrong'))
 rejects(lambda:published[c._call.signature.name](bad));mutations+=1
ok(str(mutations)+' exact-key rawparams/indices/every tensor shape,dtype,cpu,values,optional absence/report layout/bias mutations reject with canonical public UnsupportedCase')
seen=[]
f=M.wrap(lambda c:seen.append(c) or 'previous')
for c in cs:
 unknown=types.SimpleNamespace(id='unknown-'+c.id,_call=c._call,params=c.params);assert f(unknown)=='previous'
assert len(seen)==9
ok('9 unknown keys delegate original previous exactly once')
for c in cs:
 cand=M.make(c);args=[types.SimpleNamespace(shape=s.shape,dtype=s.dtype,device='cpu' if s.cpu else 'npu:0') for n in c._call.signature.inputs for s in [c._call.specs[n]]]
 rejects(lambda:cand.implementation.run(*args),RuntimeError);cand.cleanup()
ok('all 9 wrong-host actual run paths fail closed before device import')
data=M.snapshot();load=M.private_load(data)
# All original function definitions are loaded unchanged, without calling factories.
a=load(M.FP8);b=load(M.BF16);assert callable(a.fp8_bmm) and callable(b.make_candidate)
assert load(M.MATH)._MathCall.__init__.__globals__['_owned_native'] is load(M.BASE)._owned_native
parent=b._load_parent();assert callable(parent.make_binding)
ok('private full original FP8 and BF16 loader, pinned importlib parent and math bridge identities; no ordinary bridge import')
# Frozen bytes tampering in private temporary copies only.
oldroot=M.DATA_ROOT
with tempfile.TemporaryDirectory(dir=O) as td:
 import shutil
 for n in M.DATA_HASHES:shutil.copyfile(O/n,Path(td)/n)
 M.DATA_ROOT=Path(td)
 try:
  for n in M.DATA_HASHES:
   orig=(Path(td)/n).read_bytes();(Path(td)/n).write_bytes(orig+b' ')
   rejects(M.metadata,RuntimeError);(Path(td)/n).write_bytes(orig)
 finally:M.DATA_ROOT=oldroot
with tempfile.TemporaryDirectory(dir=O) as td:
 p=Path(td)/'dep';p.write_bytes(b'changed');rejects(lambda:M.checked(p,'0'*64),RuntimeError)
ok('CONTRACTS/PINS/both original FROZEN files and dependency bytes fail closed on hash changes')
# Real run lifecycle with metadata-only factory fixtures; no numerical claim.
saved={n:getattr(M,n) for n in ['admit','private_load','require_guard','environment','loaded_libraries']}
try:
 M.admit=lambda **kwargs:(data,'guard','env');M.require_guard=lambda:'guard';M.environment=lambda:'env';M.loaded_libraries=lambda:None
 for fail_run,fail_cleanup in [(False,False),(True,False),(True,True),(False,True)]:
  counts={'make':0,'run':0,'cleanup':0}
  def factory(c):
   counts['make']+=1
   def run(*a):
    counts['run']+=1
    if fail_run:raise RuntimeError('run failure')
    return 'metadata-only-result'
   def clean():
    counts['cleanup']+=1
    if fail_cleanup:raise RuntimeError('cleanup failure')
   return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=clean)
  M.private_load=lambda d:lambda p:types.SimpleNamespace(fp8_bmm=factory,make_candidate=factory)
  for c in cs:
   counts.update(make=0,run=0,cleanup=0);cand=M.make(c);assert counts['make']==0
   args=[types.SimpleNamespace(shape=s.shape,dtype=s.dtype,device='cpu' if s.cpu else 'npu:0') for n in c._call.signature.inputs for s in [c._call.specs[n]]]
   if fail_run:rejects(lambda:cand.implementation.run(*args),RuntimeError)
   else:
    assert cand.implementation.run(*args)=='metadata-only-result'
    if fail_cleanup:rejects(cand.cleanup,RuntimeError)
    else:cand.cleanup()
   cand.cleanup();assert counts==dict(make=1,run=1,cleanup=1)
finally:
 for n,v in saved.items():setattr(M,n,v)
ok('36 CPU lifecycle fixtures: factory lazy, run success/failure, cleanup success/failure, exactly-once cleanup and closed-state; no numeric execution')
# Exercise source hash mismatch through actual snapshot and actual lazy construction.
with tempfile.TemporaryDirectory(dir=O) as td:
 import shutil
 for n in M.DATA_HASHES:shutil.copyfile(O/n,Path(td)/n)
 shutil.copytree(O/'closure',Path(td)/'closure')
 item=next(iter(M.metadata()['PINS.json']['sources'].values()))
 dep=Path(td)/item['file'];dep.write_bytes(dep.read_bytes()+b'\n# changed')
 M.DATA_ROOT=Path(td)
 try:rejects(lambda:M.make(cs[0]),RuntimeError)
 finally:M.DATA_ROOT=oldroot
ok('actual make -> snapshot rejects a changed frozen source before private loader/factory')
# Ensure ambient modules cannot capture any private bridge or original importlib loader.
poison_names=['benchmarks.ascend_math_more','benchmarks.api','parent_adapter','orig_binding']
old_modules={n:sys.modules.get(n) for n in poison_names}
try:
 for n in poison_names:sys.modules[n]=types.ModuleType('poison')
 private=M.private_load(data)
 assert private(M.MATH)._MathCall.__init__.__globals__['_owned_native'] is private(M.BASE)._owned_native
 assert private(M.BF16)._load_parent().make_binding.__code__.co_filename in data
 assert all(sys.modules[n].__name__=='poison' for n in poison_names)
finally:
 for n,v in old_modules.items():
  if v is None:sys.modules.pop(n,None)
  else:sys.modules[n]=v
ok('poisoned ambient bridge/API/importlib target names bypassed without sys.modules writes')
for c in cs:
 bad=types.SimpleNamespace(id=c.id,_call=copy.deepcopy(c._call),params=copy.deepcopy(c.params))
 cand=M.make(bad);bad.params['bias']=True
 rejects(lambda:cand.implementation.run());cand.cleanup()
 args=[types.SimpleNamespace(shape=t.shape,dtype=t.dtype,device='cpu' if t.cpu else 'npu:0') for n in c._call.signature.inputs for t in [c._call.specs[n]]]
 for index in range(len(args)):
  changed=list(args);changed[index]=types.SimpleNamespace(shape=(1,),dtype=args[index].dtype,device=args[index].device)
  rejects(lambda:M.validate_inputs(c,changed,{}))
 rejects(lambda:M.validate_inputs(c,args,{'bias':object()}))
ok('postconstruction Case mutation and actual input shape/extra bias rejects with public UnsupportedCase')

# Parent regression: private native objects cannot outlive their original guard.
saved={n:getattr(M,n) for n in ['admit','private_load','require_guard','environment','loaded_libraries']}
try:
 identity=['first'];counts={'build':0,'run':0,'cleanup':0};admissions=[]
 M.require_guard=lambda:identity[0];M.environment=lambda:'env';M.loaded_libraries=lambda:None
 def admission(runtime=True):
  admissions.append(runtime);return data,identity[0],'env'
 M.admit=admission
 def factory(c):
  counts['build']+=1
  def run(*args):counts['run']+=1;return 'cpu-fixture'
  def cleanup():counts['cleanup']+=1
  return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=cleanup)
 M.private_load=lambda d:lambda p:types.SimpleNamespace(fp8_bmm=factory,make_candidate=factory)
 c=cs[0];candidate=M.make(c)
 args=[types.SimpleNamespace(shape=s.shape,dtype=s.dtype,device='cpu' if s.cpu else 'npu:0') for n in c._call.signature.inputs for s in [c._call.specs[n]]]
 candidate.implementation.run(*args);candidate.implementation.run(*args)
 assert admissions==[True,False] and counts['build']==1
 identity[0]='different-guard'
 rejects(lambda:candidate.implementation.run(*args),RuntimeError);rejects(candidate.cleanup,RuntimeError)
 assert counts['run']==2 and counts['cleanup']==0
 identity[0]='first';candidate.cleanup();candidate.cleanup();assert counts['cleanup']==1
finally:
 for n,v in saved.items():setattr(M,n,v)
ok('original guard lifetime enforced for reuse and cleanup; no native call under changed guard; full runtime hashes once per Candidate')

assert not blocked and not dl
result=dict(passed=True,checks=checks,exact_keys=9,mutations=mutations,device_import_attempts=blocked,native_dlopen_attempts=dl,npu_runs=0,numerical_runs=0,accepted_added=0,shared_provider_writes=0)
(O/'PARENT_CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
