import sys,builtins,types,ctypes,json,dataclasses,copy,tempfile,hashlib,ast
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O))
original=builtins.__import__;blocked=[]
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):
  blocked.append(name);raise AssertionError('Device import forbidden: '+name)
 return original(name,*a,**kw)
builtins.__import__=protected
def audit(event,args):
 if event=='ctypes.dlopen':raise AssertionError('Native loading forbidden')
sys.addaudithook(audit)
import build_contracts as B,registration as M
from benchmarks.ascend_baselines import Candidate,UnsupportedCase
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
c,=B.cases();checks=[]
def reject(fn,kind=UnsupportedCase):
 try:fn()
 except kind as e:
  assert type(e) is kind;return
 raise AssertionError('Expected '+kind.__name__)
# Exact real registry function body runs with a private module namespace resolving
# a copied provider dictionary. Shared registry and provider globals stay read-only.
shared_before=dict(PROVIDERS);private=dict(PROVIDERS);M.install(private)
regns=dict(registry.candidate_factory.__globals__)
real_import=regns['import_module']
regns['import_module']=lambda name: types.SimpleNamespace(PROVIDERS=private) if name=='benchmarks.ascend_all_pairs.ops_transformer' else real_import(name)
formal=types.FunctionType(registry.candidate_factory.__code__,regns)
hashes=[];checked=M.checked
M.checked=lambda p,h:(hashes.append(str(p)),checked(p,h))[1]
cand=formal(c,'ops-transformer')()
assert isinstance(cand,Candidate) and cand.implementation.run.__code__.co_filename==str(O/'registration.py')
assert 'mkldnn.enabled=False' in cand.adaptation
assert all(hashes.count(str(O/item['file']))==1 for item in M.metadata()['PINS.json']['sources'].values())
cand.cleanup();assert PROVIDERS==shared_before
M.checked=checked
checks.append('Real manifest case -> exact registry function body -> copied actual provider routes -> real new Candidate; 8 source hashes once; no mock candidate or return value')
n=0
for field in ('indices','params'):
 for name in getattr(c._call,field):
  bad=B.Case(c.id,dataclasses.replace(c._call,**{field:dict(getattr(c._call,field),**{name:'wrong'})}),None)
  reject(lambda:formal(bad,'ops-transformer')());n+=1
for name,s in c._call.specs.items():
 if s is None:continue
 for delta in ({'shape':(1,)},{'dtype':'float64'},{'cpu':not s.cpu},{'values':[999]}):
  specs=dict(c._call.specs);specs[name]=dataclasses.replace(s,**delta)
  bad=B.Case(c.id,dataclasses.replace(c._call,specs=specs),None);reject(lambda:formal(bad,'ops-transformer')());n+=1
bad=types.SimpleNamespace(id=c.id,_call=c._call,params=dict(c.params,bias=True));reject(lambda:formal(bad,'ops-transformer')());n+=1
contract_rejections=n
checks.append(f'{n} same-key contract mutations reject canonical public UnsupportedCase')
seen=[];previous=lambda x:seen.append(x) or object();wrapped=M.wrap(previous)
u=types.SimpleNamespace(id='unknown',_call=c._call,params=c.params);wrapped(u);assert seen==[u]
reject(lambda:M.wrap(None)(u));checks.append('Unknown key delegates previous once; absent previous public UnsupportedCase')
args=[types.SimpleNamespace(shape=s.shape,dtype=s.dtype,device='cpu' if s.cpu else 'npu:0') for name in c._call.signature.inputs for s in [c._call.specs[name]]]
cand=M.make(c);reject(lambda:cand.implementation.run(*args),RuntimeError);cand.cleanup();reject(lambda:cand.implementation.run(*args),RuntimeError)
checks.append('Actual lazy run refuses unsupported host/runtime environment before torch/native imports; cleanup idempotent and closed state enforced')
data=M.snapshot();load=M.private_load(data)
assert callable(load(M.ADAPTER).candidate) and callable(load(M.BINDING).dense_fp32)
assert load(M.MATH)._MathCall.__init__.__globals__['_owned_native'] is load(M.BASE)._owned_native
checks.append('Original complete adapter/binding/math/base definitions loaded unchanged through private import bridge; original factory not numerically invoked')
from backend_context import evaluation_backend
for initial in (True,False):
 for fail in (False,True):
  t=types.SimpleNamespace(backends=types.SimpleNamespace(mkldnn=types.SimpleNamespace(enabled=initial)),bmm=object(),__version__='CPU fixture',get_num_threads=lambda:4)
  events=[];orig=t.bmm
  try:
   with evaluation_backend(t,events.append):
    assert t.backends.mkldnn.enabled is False and t.bmm is orig
    if fail:raise ValueError('checker error')
  except ValueError:assert fail
  assert t.backends.mkldnn.enabled==initial and t.bmm is orig and events[-1]['phase']=='restored'
checks.append('Explicit backend context restores True/False on normal/error exits; torch.bmm identity unchanged; CPU fixtures only')
# Verify backend validation independently; no device import or numeric workload.
fake=types.SimpleNamespace(backends=types.SimpleNamespace(mkldnn=types.SimpleNamespace(enabled=True)),__version__='2.10.0+cpu',get_num_threads=lambda:4)
def fakeimport(name,*a,**kw):return fake if name=='torch' else protected(name,*a,**kw)
builtins.__import__=fakeimport
try:
 reject(M.require_backend,RuntimeError);assert fake.backends.mkldnn.enabled is True
 fake.backends.mkldnn.enabled=False;M.require_backend()
finally:builtins.__import__=protected
checks.append('Candidate backend validation rejects default True without mutation and accepts explicit False CPU metadata fixture')
# Tampering tests touch only disposable files in this private report.
import shutil
oldroot=M.DATA_ROOT
with tempfile.TemporaryDirectory(dir=O) as td:
 td=Path(td)
 for name in M.DATA_HASHES:shutil.copyfile(O/name,td/name)
 shutil.copytree(O/'closure',td/'closure')
 M.DATA_ROOT=td
 try:
  for name in M.DATA_HASHES:
   path=td/name;raw=path.read_bytes();path.write_bytes(raw+b' ')
   reject(M.metadata,RuntimeError);path.write_bytes(raw)
  item=next(iter(M.metadata()['PINS.json']['sources'].values()));path=td/item['file'];path.write_bytes(path.read_bytes()+b'\n# altered')
  reject(lambda:M.make(c),RuntimeError)
 finally:M.DATA_ROOT=oldroot
checks.append('All 4 pinned metadata documents and actual source-byte tampering fail closed before native construction')
# Metadata-only lifecycle fixtures are separate from real construction proof.
saved={n:getattr(M,n) for n in ['admit','private_load','require_guard','environment','loaded_libraries','require_backend']}
try:
 identity=['first'];counts={};admissions=[]
 M.require_guard=lambda:identity[0];M.environment=lambda:'env';M.loaded_libraries=lambda:None;M.require_backend=lambda:None
 def admit(data,runtime=True):admissions.append(runtime);return data,identity[0],'env'
 M.admit=admit
 for fail in (False,True):
  counts.update(build=0,run=0,cleanup=0);admissions.clear();identity[0]='first'
  def factory(case,**kw):
   counts['build']+=1
   def run(*args):
    counts['run']+=1
    if fail:raise RuntimeError('CPU lifecycle fixture exception')
    return args
   def cleanup():counts['cleanup']+=1
   return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=cleanup)
  M.private_load=lambda data:lambda path:types.SimpleNamespace(candidate=factory,dense_fp32=object())
  cand=M.make(c)
  if fail:
   reject(lambda:cand.implementation.run(*args),RuntimeError);assert counts==dict(build=1,run=1,cleanup=1)
  else:
   assert cand.implementation.run(*args)==tuple(args)
   assert cand.implementation.run(*args)==tuple(args)
   assert admissions==[True,False] and counts['build']==1
   identity[0]='other';reject(lambda:cand.implementation.run(*args),RuntimeError)
   assert counts['run']==2 and counts['cleanup']==0
   identity[0]='first';cand.cleanup();assert counts['cleanup']==1
  cand.cleanup();assert counts['cleanup']==1
finally:
 for n,v in saved.items():setattr(M,n,v)
checks.append('Separate CPU lifecycle fixtures: one native construction/runtime-hash admission per candidate, exception cleanup once, cross-guard reuse fails closed, original-guard cleanup required')
assert not blocked and not any(n in sys.modules for n in ('torch','torch_npu','acl','tilelang','tvm'))
result=dict(passed=True,checks=checks,exact_real_cases=1,contract_rejections=contract_rejections,device_imports=0,native_loads=0,NPU_runs=0,numerical_runs=0,shared_registry_changed=False,accepted_added=0)
(O/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
