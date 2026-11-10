import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original_import=builtins.__import__; forbidden=[]; native_load_attempts=[]
import ctypes
def audit(event,args):
 if event == 'ctypes.dlopen':
  native_load_attempts.append(str(args));raise AssertionError('Native library load forbidden in CPU checks')
sys.addaudithook(audit)
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
M=types.ModuleType('registration');M.__file__=str(O/'integration700_fp16_bmm_registration1.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):checks.append(x)
def reject(fn):
 try:fn()
 except (RuntimeError,ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
# Instantiate original CPU-only manifest Calls, then execute exactly the passive
# Case/Implementation declarations from original API (its module imports torch).
import ast, dataclasses, functools
from typing import Any, Callable, Optional
sys.path.insert(0,'/root/dyq/workspace/TileOPs/src')
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate,Call
from tileops.manifest.values import ADTValue
api_path=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py')
tree=ast.parse(api_path.read_bytes()); nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('Case','Implementation','_report_value')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(api_path),'exec'),globals())
cs=[]
for op in ('BmmFwdOp',):
 plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
 for row in load_workloads(op):
  for dtype in row.get('dtype_cases') or [{}]:
   call=instantiate(plan,row,dtype)
   if ('ops-transformer',op,call.case_id) in M.CASE_PARAMS:cs.append(Case(call.case_id,call,None))
assert len(cs)==1
passed('one real manifest Call instances and exact original passive Case declarations; no workload or inputs accessed')
def case(k,p):
 c=next(c for c in cs if c._call.signature.name==k[1])
 return types.SimpleNamespace(id=k[2],params=p,_call=c._call)
for c in cs:
 assert M.matches(c); candidate=M.make(c); assert callable(candidate.implementation.run)
 assert not M.matches(c,'wrong-baseline')
 bad=case(('ops-transformer',c._call.signature.name,c.id),dict(c.params,unexpected=True));assert not M.matches(bad);reject(lambda:M.make(bad))
passed('1 real canonical params CPU construction; wrong baseline/full params rejected')
previous=[]
def old(c):previous.append(c);return 'old'
f=M.wrap(old)
for c in cs:
 unknown=case(('ops-transformer',c._call.signature.name,'old-'+c.id),c.params);assert f(unknown)=='old'
assert len(previous)==1;passed('1 previous-route mockfactory calls preserved; unknown keys delegated')
# Native dependencies must be private even when all names are poisoned.
poison={n:sys.modules.get(n) for n in ('binding','resource_gate','benchmarks.ascend_math_more','adapter','benchmarks.api')}
try:
 for n in poison:sys.modules[n]=types.ModuleType('poison')
 private=M._load(M._snapshot())
 assert private['binding'].dense_fp32.__globals__['__file__']==str(M.ROOT/'binding.py')
 assert private['benchmarks.ascend_math_more']._MathCall.__init__.__globals__['_owned_native'] is private['benchmarks.ascend_baselines']._owned_native
 assert all(sys.modules[n].__name__=='poison' for n in poison)
finally:
 for n,v in poison.items():
  if v is None:sys.modules.pop(n,None)
  else:sys.modules[n]=v
passed('private bytes load ignores same-name sys.modules and leaves sys.modules untouched')
# Read-only real mapping helper: no library loading or native calls.
p=Path(M.REQUIRED_LOADED[0]);digest=M.LOADED_VENDOR[str(p.resolve())]
with p.open('rb') as src,mmap.mmap(src.fileno(),0,access=mmap.ACCESS_READ) as mm:
 fields=next(l.split(maxsplit=5) for l in Path('/proc/self/maps').read_text().splitlines() if l.endswith(str(p)))
 assert M._overlay_mapping_matches(str(p),digest,fields)
 assert not M._overlay_mapping_matches(str(p),'0'*64,fields)
 wrong=fields[:];wrong[3]='ff:ff';assert not M._overlay_mapping_matches(str(p),digest,wrong)
 wrong=fields[:];wrong[4]=str(int(fields[4])+1);assert not M._overlay_mapping_matches(str(p),digest,wrong)
passed('real readonly mmap bytes plus kernel device/inode match; wrong hash/device rejected')
# Tamper tests only own temporary files, never original evidence.
with tempfile.TemporaryDirectory(dir=O) as td:
 p=Path(td)/'modified';p.write_text('changed')
 reject(lambda:M._checked(p,M.PINS[str(M.ROOT/'binding.py')]))
 old_manifest=M.MANIFEST;M.MANIFEST=str(p)
 reject(M._snapshot);M.MANIFEST=old_manifest
passed('binding-byte mismatch and manifest mutation fail closed')
assert M._guard_identity() is None;reject(M._require_guard)
passed('current unguarded CPU worker cannot acquire admission')
# Wrapper state-machine tests use mocks; these assert no numeric/ABI result.
saved={n:getattr(M,n) for n in ('_snapshot','_guard_identity','_require_guard','_environment','_prepare_admission','_loaded_vendor_pins')}
frozen_for_mock=M._frozen_case(M._key(cs[0]));original_frozen=M._frozen_case;M._frozen_case=lambda key:frozen_for_mock
events=[];identity=('guard',1);environment=('env',1);run_dir=O/'mock_run';run_dir.mkdir(exist_ok=True)
def admission(k):
 events.append('admission');return ({'binding':types.SimpleNamespace(dense_fp32=build),'adapter':types.SimpleNamespace(candidate=adapt)},run_dir,identity,environment)
def adapt(c,*,native_factory):
 assert dict(c._call.indices)==dict(cs[0]._call.indices)
 return native_factory(c)
def build(c):
 events.append('build')
 return types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda *a,**kw:events.append('call')),cleanup=lambda:events.append('cleanup'))
try:
 M._snapshot=lambda runtime=False:events.append('pins')
 M._guard_identity=lambda:identity;M._require_guard=lambda:identity;M._environment=lambda:environment
 M._prepare_admission=admission;M._loaded_vendor_pins=lambda require=False:None
 c=cs[0];cand=M.make(c);events.append('inputs');events.append('reference');cand.implementation.run()
 assert events.index('admission')<events.index('inputs')<events.index('reference')<events.index('build')
 assert events.count('admission')==1 and events[-1]=='cleanup'
 cand.implementation.run();assert events.count('admission')==2
 passed('pre-input constructor admission; no full-envelope recharge after reference; token consumed once')
 for kind in ('guard','env','pins'):
  cand=M.make(c);events.clear()
  if kind=='guard':M._require_guard=lambda:('changed',)
  if kind=='env':M._environment=lambda:('changed',)
  if kind=='pins':M._snapshot=lambda runtime=False:(_ for _ in ()).throw(RuntimeError('changed pins'))
  reject(cand.implementation.run);assert 'build' not in events
  M._require_guard=lambda:identity;M._environment=lambda:environment;M._snapshot=lambda runtime=False:events.append('pins')
 passed('guard/environment/dependency change rejects before native construction')
 cand=M.make(c);events.clear()
 def badmap(require=False):
  if require:raise RuntimeError('bad maps')
 M._loaded_vendor_pins=badmap;reject(cand.implementation.run);assert events[-1]=='cleanup'
 passed('cleanup also runs when required map check fails after native construction')
 M._loaded_vendor_pins=lambda require=False:None
 def broken_adapter(c,*,native_factory):
  native_factory(c);raise RuntimeError('adapter construction failed')
 good_adapt=adapt;adapt=broken_adapter
 cand=M.make(c);events.clear();reject(cand.implementation.run);assert events[-1]=='cleanup'
 adapt=good_adapt
 def throwing_build(c):
  child=build(c)
  child.implementation.run=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('native run failed'))
  return child
 normal_build=build
 # Wrap original without recursive name lookup.
 def failing_build(c):
  child=normal_build(c);child.implementation.run=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('native run failed'));return child
 build=failing_build;cand=M.make(c);events.clear();reject(cand.implementation.run);assert events[-1]=='cleanup';build=normal_build
 passed('cleanup on adapter construction failure and native execution failure')
 M._loaded_vendor_pins=lambda require=False:None;M._guard_identity=lambda:None
 events.clear();cand=M.make(c);assert 'admission' not in events;cand.implementation.run();assert events.count('admission')==1
 passed('CPU construction is lazy; run with no preadmission requests fresh admission')
finally:
 for n,v in saved.items():setattr(M,n,v)
M._frozen_case=original_frozen
# Test actual environment validation without importing providers.
original_env=dict(os.environ);prefix=sys.prefix
try:
 os.environ.update(M.ENV);os.environ.update(LD_LIBRARY_PATH=M.LD_PATH,PYTHONPATH=M.PYTHON_PATH,FP16_BMM_RUN_ROOT=str(O/'environment-test'))
 for n in M.CACHE_NAMES:os.environ[n]=str(O/'environment-test'/n)
 sys.prefix='/root/dyq/workspace/.venv';M._environment()
 os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']='/wrong';reject(M._environment)
finally:os.environ.clear();os.environ.update(original_env);sys.prefix=prefix
passed('actual environment validator rejects changed vendor; no device imports')
# Same params but altered indices cannot enter; snapshot recursively immutable.
for c in cs:
 bad=types.SimpleNamespace(id=c.id,params=c.params,_call=types.SimpleNamespace(signature=c._call.signature,indices=dict(c._call.indices,T='bfloat16'),tensors=c._call.tensors))
 assert not M.matches(bad);reject(lambda:M.make(bad))
 frozen=M._frozen_case(M._key(c))
 assert dict(frozen._call.indices)==dict(c._call.indices)
 try:frozen._call.indices['T']='bfloat16'
 except TypeError:pass
 else:raise AssertionError('mutable snapshot')
passed('all one resolved indices bind independently of report params; immutable adapter snapshots')
assert not forbidden
(run_dir/'loaded-maps.txt').unlink(missing_ok=True)
run_dir.rmdir()
# Exercise real kernel flock and /proc identities using an isolated CPU fixture.
# Never opens, locks or changes the shared device lock in this fixture.
import subprocess
with tempfile.TemporaryDirectory(dir=O) as td:
 lock=Path(td)/'cpu-fixture.lock';lock.write_text('{}')
 holder=subprocess.Popen([sys.executable,'-S','-c',
  "import fcntl,sys; f=open(sys.argv[1],'r'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); sys.stdin.read()",str(lock)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 try:
  assert holder.stdout.readline().strip()=='held'
  group=os.getpgrp();ticks,_=M._proc(group)
  meta={'child':{'pid':group,'start_ticks':ticks},'owner':'cpu-fixture-only','phase':'running','device':0,
    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':holder.pid,'start_ticks':M._proc(holder.pid)[0],'requested_at':1}
  oldpath=M.Path;oldowner=os.environ.get('FP16_BMM_GUARD_OWNER')
  M.Path=lambda p:lock if str(p)=='/run/lock/ascend-npu/device-0.lock' else Path(p)
  os.environ['FP16_BMM_GUARD_OWNER']='cpu-fixture-only'
  lock.write_text(json.dumps(meta));assert M._guard_identity() is not None
  for field,value in [('boot_id','wrong'),('start_ticks',-1),('owner','wrong'),('phase','starting')]:
   changed=dict(meta);changed[field]=value;lock.write_text(json.dumps(changed));assert M._guard_identity() is None
  changed=dict(meta,child={'pid':group,'start_ticks':-1});lock.write_text(json.dumps(changed));assert M._guard_identity() is None
  lock.write_text(json.dumps(meta));holder.communicate('');assert M._guard_identity() is None
 finally:
  M.Path=oldpath
  if oldowner is None:os.environ.pop('FP16_BMM_GUARD_OWNER',None)
  else:os.environ['FP16_BMM_GUARD_OWNER']=oldowner
  if holder.poll() is None:holder.communicate('')
passed('isolated real kernel flock plus actual boot/startticks/process-group validation; stale metadata and released lock rejected')
# Original approved adapter.candidate executes with original passive Implementation;
# only tensor API tokens and native factory are structural mocks, never numeric.
private=M._load(M._snapshot());adapter=private['adapter'];runner=adapter.make_runner
seen=[]
def metadata_runner(op,params,*,native_factory,tensor_api=None):
 seen.append((op,dict(params)))
 return runner(op,params,native_factory=native_factory,tensor_api=types.SimpleNamespace(float16='float16',int32='int32'))
adapter.make_runner=metadata_runner
for c in cs:
 def fake_native(proxy):
  assert proxy._call.signature.name in ('BmmFwdOp','GemmFwdOp')
  return private['benchmarks.ascend_baselines'].Candidate(types.SimpleNamespace(run=lambda *a:None),'mock','mock',{},cleanup=lambda:None)
 cand=adapter.candidate(M._frozen_case(M._key(c)),native_factory=fake_native)
 assert callable(cand.implementation.run)
 assert seen[-1]==(c._call.signature.name,dict(c._call.indices))
passed('all one original adapter.candidate paths accept immutable real-Call snapshots with exact scalar/tuple/dtype indices; no math or native loading')
# Prove fail-before-import ordering on actual admission host/disk failure.
saved={n:getattr(M,n) for n in ('_require_guard','_environment','_snapshot','_load','_loaded_vendor_pins','_new_run')}
try:
 M._require_guard=lambda:('fixture',);M._environment=lambda:('fixture',);M._snapshot=lambda runtime=False:{}
 M._loaded_vendor_pins=lambda require=False:None
 with tempfile.TemporaryDirectory(dir=O) as td:
  M._new_run=lambda key:Path(td)
  M._load=lambda data:{'resource_gate':types.SimpleNamespace(capacity=lambda:{'effective_available_host_bytes':0})}
  reject(lambda:M._prepare_admission(M._key(cs[0])))
finally:
 for n,v in saved.items():setattr(M,n,v)
passed('actual preinput admission refuses insufficient effective host capacity before any device import')

# Real Call mutations must reject with precisely the canonical public exception.
from benchmarks.ascend_baselines import UnsupportedCase
for field,value in [('indices',dict(cs[0]._call.indices,T='bfloat16')),('params',{'unexpected':True}),('indices',dict(cs[0]._call.indices,B=1))]:
 call=dataclasses.replace(cs[0]._call,**{field:value});bad=Case(cs[0].id,call,None)
 assert not M.matches(bad)
 try:M.wrap(None)(bad)
 except UnsupportedCase as exc:assert type(exc) is UnsupportedCase
 else:raise AssertionError('Canonical rejection missing')
for name in ('a','b','d'):
 specs=dict(cs[0]._call.specs);specs[name]=dataclasses.replace(specs[name],dtype='bfloat16')
 bad=Case(cs[0].id,dataclasses.replace(cs[0]._call,specs=specs),None)
 assert not M.matches(bad)
private=M._load(M._snapshot());assert private['benchmarks.ascend_baselines'].UnsupportedCase is UnsupportedCase
passed('real Call dtype/batch/rawparams/tensor mutations rejected via public UnsupportedCase, including wrapped exact route')
frozen=M._frozen_case(M._key(cs[0]));assert isinstance(frozen._call,Call)
assert M.matches(frozen) and frozen._call.present('a') and not frozen._call.present('out')
for fn in (lambda:frozen._call.params.__setitem__('extra',1),lambda:frozen._call.specs.__setitem__('a',None),lambda:setattr(frozen,'id','wrong')):
 try:fn()
 except (AttributeError,TypeError):pass
 else:raise AssertionError('Mutable frozen contract')
passed('original Case AST and true Call, frozen report/rawparams/indices/specs/presence verified')
# Execute exact original native nested function AST with metadata-only tensor API.
# No imports, tensor allocation, library loads, numeric computations, or ABI claims.
import ctypes as C
binding_tree=ast.parse(Path(M.LOCAL_FILES[3]).read_bytes())
outer=next(n for n in binding_tree.body if isinstance(n,ast.FunctionDef) and n.name=='dense_fp32')
funcs=[n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name in ('tensorlist','run','cleanup')]
trace=[]
class MetaTensor:
 def __init__(self,shape,dtype='float16'):self.shape=shape;self.dtype=dtype;self.device='npu:0'
 def float(self):trace.append(('widen',self.shape,self.dtype));return MetaTensor(self.shape,'float32')
 def contiguous(self):return self
 def __iter__(self):return iter([MetaTensor(self.shape[1:],self.dtype) for _ in range(self.shape[0])])
 def to(self,dtype):trace.append(('cast',dtype));return MetaTensor(self.shape,dtype)
class Base:
 def aclCreateTensorList(self,arr,n):trace.append(('tensorlist',n));return len(trace)
 def aclDestroyTensorList(self,p):trace.append(('destroy',p));return 0
class Math:
 def record(self):return {'keep':[],'tensors':[]}
 def tensor(self,t,r):r['tensors'].append(len(r['keep'])+100);return r['tensors'][-1]
 def call(self,args,r,device):trace.append(('native',len(args),device));assert len(args)==20
 def check(self,r,label):assert r==0
 def cleanup(self):trace.append(('cleanup',))
api=types.SimpleNamespace(empty=lambda shape,**kw:MetaTensor(shape,kw['dtype']),stack=lambda ys:MetaTensor((len(ys),)+ys[0].shape,ys[0].dtype),npu=types.SimpleNamespace(synchronize=lambda:trace.append(('sync',))))
ns={'C':C,'b':Math(),'base':Base(),'lists':[],'case':cs[0],'torch':api}
exec(compile(ast.Module(body=funcs,type_ignores=[]),M.LOCAL_FILES[3],'exec'),ns)
y=ns['run'](MetaTensor((8,2048,2048)),MetaTensor((8,2048,2048)));ns['cleanup']()
assert y.shape==(8,2048,2048) and y.dtype=='float16'
assert [x for x in trace if x[0]=='tensorlist']==[('tensorlist',8)]*3
assert len([x for x in trace if x[0]=='native'])==1 and len([x for x in trace if x[0]=='widen'])==2
assert len([x for x in trace if x[0]=='destroy'])==3 and trace[-1]==('cleanup',)
passed('unchanged original binding run/tensorlist/cleanup AST: full batch 8, three lists, one V5 call, FP32 widening and original FP16 cast, synchronized cleanup; metadata only')
# Actual adapter bmm invokes native product once with complete inputs.
adapter=private['adapter'];seen=[]
run,inner=adapter.make_runner('BmmFwdOp',cs[0]._call.indices,native_factory=lambda proxy:types.SimpleNamespace(implementation=types.SimpleNamespace(run=lambda a,b:seen.append((a,b)) or MetaTensor((8,2048,2048)))),tensor_api=types.SimpleNamespace(float16='float16',int32='int32'))
a=MetaTensor((8,2048,2048));b=MetaTensor((8,2048,2048));assert run(a,b).shape==(8,2048,2048) and seen==[(a,b)]
passed('actual FP16 adapter make_runner passes complete 8 batch inputs once unchanged')
# Admission's original code with only its device import removed is tested with
# capacity metadata. No device module is imported, and no memory is allocated.
source=ast.parse(Path(M.__file__).read_bytes())
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_admission')
node.body=[n for n in node.body if not (isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))]
with tempfile.TemporaryDirectory(dir=O) as td:
 ns=dict(M.__dict__);ns.update(_require_guard=lambda:('guard',),_environment=lambda:('env',),_snapshot=lambda runtime=False:{},_load=lambda data:{'resource_gate':types.SimpleNamespace(capacity=lambda:{'effective_available_host_bytes':10**12})},_loaded_vendor_pins=lambda:None,_new_run=lambda _:Path(td),torch=types.SimpleNamespace(npu=types.SimpleNamespace(mem_get_info=lambda:(10**12,10**12))))
 exec(compile(ast.Module(body=[node],type_ignores=[]),M.__file__,'exec'),ns)
 ns['_prepare_admission'](M._key(cs[0]))
 admission=json.loads((Path(td)/'RESOURCE_ADMISSION.json').read_text())
 assert admission['stage']=='before_input_allocation' and admission['input_elements']==67108864 and admission['output_elements']==33554432
 assert admission['workspace_allowance']==4294967296 and admission['params']==json.loads(M.CASE_PARAMS[M._key(cs[0])])
passed('original admission AST with device import removed serializes full immutable batch envelope; metadata-only HBM fixture')
historical=json.loads((O.parent/'fp16_local_remaining3/square-2k-float16/result/RUNTIME.json').read_text())['environment']
assert all(M.ENV[n]==v for n,v in historical.items() if n in M.ENV)
assert M.LD_PATH==historical['LD_LIBRARY_PATH']
passed('startup environment including .pth vendor prefix equals actual successful FP16 RUNTIME receipt')

# Full real pin check is CPU file hashing, not a mocked runtime validation.

M._snapshot(runtime=True)
passed('all frozen runtime/source/dependency bytes rehashed unchanged')
checker=Path('/root/dyq/workspace/TileOPs/scripts/ascend_pair_check.py').read_text()
assert checker.index('candidate = candidate_factory(case, args.library)()') < checker.index('impl_args = case.inputs')
passed('actual canonical checker constructs Candidate before first inputs access')
assert not forbidden and not native_load_attempts
result={'passed':True,'checks':checks,'device_import_attempts':forbidden,'npu_runs':0,'numerical_runs':0,'native_library_loads':len(native_load_attempts),'native_load_attempts':native_load_attempts,'mock_numeric_claims':False}
(O/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
