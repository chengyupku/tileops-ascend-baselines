import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path[:0]=['/root/dyq/workspace/TileOPs','/root/dyq/workspace/TileOPs/src','/root/dyq/workspace/.venv/lib/python3.12/site-packages']
original_import=builtins.__import__; forbidden=[]
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
M=types.ModuleType('registration');M.__file__=str(O/'integration700_bf16_conv_registration1.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):checks.append(x)
def reject(fn):
 try:fn()
 except (RuntimeError,ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
# Instantiate the actual manifest Call, and the unmodified Case class AST.
# This excludes only api module imports which would import torch at module scope.
import ast,dataclasses,functools,typing,copy
from tileops.manifest import load_manifest,load_adts,load_workloads
from tileops.manifest.plan import entry_plan
from tileops.manifest.workload import instantiate
from tileops.manifest.values import ADTValue
op='Conv1dFwdOp'
plan=entry_plan(op,load_manifest()[op],load_adts(),resolve=False)
realcall=next(c for c in (instantiate(plan,r,d) for r in load_workloads(op) for d in r.get('dtype_cases') or [{}]) if c.case_id=='whisper-conv1-bias-bfloat16')
api_source=Path('/root/dyq/workspace/TileOPs/benchmarks/api.py').read_bytes()
tree=ast.parse(api_source)
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Case','_report_value')]
ns=dict(dataclasses=dataclasses,functools=functools,Any=typing.Any,Optional=typing.Optional,Callable=typing.Callable,Call=type(realcall),ADTValue=ADTValue)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original-api-CPU-classes','exec'),ns)
realcase=ns['Case'](realcall.case_id,realcall,None)
assert isinstance(realcase.params['input'][0],tuple)
def case(k,p):
 return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1]),indices=dict(realcall.indices),params=dict(realcall.params),tensors=dict(realcall.tensors),present=realcall.present))
cs=[realcase]
for c in cs:
 assert M.matches(c); candidate=M.make(c); assert callable(candidate.implementation.run)
 assert not M.matches(c,'wrong-baseline')
 bad=case(('ops-transformer','Conv1dFwdOp',c.id),dict(c.params,unexpected=True));assert not M.matches(bad);reject(lambda:M.make(bad))
# Exact indices, raw parameter fields, present API, dtype and post-construction mutation.
for field in ('L_out','T','stride'):
 bad=case(('ops-transformer',op,realcase.id),dict(realcase.params))
 bad._call.indices[field]='wrong'; assert not M.matches(bad);reject(lambda:M.make(bad))
bad=case(('ops-transformer',op,realcase.id),dict(realcase.params));bad._call.present=lambda n:False
assert not M.matches(bad)
bad=case(('ops-transformer',op,realcase.id),dict(realcase.params));bad._call.params['stride']=2
assert not M.matches(bad)
mut=case(('ops-transformer',op,realcase.id),dict(realcase.params));cand=M.make(mut);mut._call.indices['L_out']=1
reject(cand.implementation.run)
frozen=M._freeze_case(realcase)
assert frozen._call.present('bias') and frozen._call.tensors['input']==((1,80,3000),'bfloat16')
try:frozen._call.indices['N']=2
except TypeError:pass
else:raise AssertionError('mutable frozen indices')
passed('real Call derived/scalar/dtype fields, presence, raw params and subsequent mutation checked; immutable compatible snapshot')
passed('1 real canonical params CPU construction; wrong baseline/full params rejected')
previous=[]
def old(c):previous.append(c);return 'old'
f=M.wrap(old)
for c in cs:
 unknown=case(('ops-transformer','Conv1dFwdOp','old-'+c.id),c.params);assert f(unknown)=='old'
assert len(previous)==1
other=case(('ops-transformer','OtherOp',realcase.id),realcase.params);assert f(other)=='old'
wrong=case(('ops-transformer',op,realcase.id),dict(realcase.params,stride=2));before=len(previous);reject(lambda:f(wrong));assert len(previous)==before
passed('1 previous-route mockfactory calls preserved; unknown keys delegated')
# Native dependencies must be private even when all names are poisoned.
poison={n:sys.modules.get(n) for n in ('binding','resource_gate','benchmarks.ascend_math_more','benchmarks.ascend_baselines','adapter','benchmarks.api')}
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
# Exercise the unmodified approved adapter.candidate with only make_runner mocked;
# this binds actual indices/presence and private Implementation without torch.
a=private['adapter']; saved_runner=a.make_runner;bound=[]
def runner_mock(op,p,*,native_factory):
 bound.append((op,p))
 return (lambda:None),native_factory(None)
a.make_runner=runner_mock
try:
 inner=private['benchmarks.ascend_baselines'].Candidate(private['benchmarks.api'].Implementation(lambda:None),'test','test',{})
 out=a.candidate(M._freeze_case(realcase),native_factory=lambda c:inner)
 assert bound==[(op,dict(M.CASE_INDICES,bias=True))]
 assert type(out.implementation) is private['benchmarks.api'].Implementation
finally:a.make_runner=saved_runner
passed('approved adapter candidate consumes exact immutable call indices/present; private API class; no tensor math')
# Read-only real mapping helper: no library loading or native calls.
p=Path(M.REQUIRED_LOADED[0]);digest=M.LOADED_VENDOR[str(p.resolve())]
with p.open('rb') as src,mmap.mmap(src.fileno(),0,access=mmap.ACCESS_READ) as mm:
 fields=next(l.split(maxsplit=5) for l in Path('/proc/self/maps').read_text().splitlines() if l.endswith(str(p)))
 assert M._overlay_mapping_matches(str(p),digest,fields)
 assert not M._overlay_mapping_matches(str(p),'0'*64,fields)
 wrong=fields[:];wrong[3]='ff:ff';assert not M._overlay_mapping_matches(str(p),digest,wrong)
 wrong=fields[:];wrong[4]='0';assert not M._overlay_mapping_matches(str(p),digest,wrong)
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
# Real kernel flock and real proc boot/startticks/pgrp, using an own-directory
# fixture lock. Never open the host NPU lock for writing or acquire that lock.
import fcntl
with tempfile.TemporaryDirectory(dir=O) as td:
 lock=Path(td)/'lock';pid=os.getpid();group=os.getpgrp()
 meta=dict(owner='conv-cpu-fixture',phase='running',device=0,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),pid=pid,start_ticks=M._proc(pid)[0],requested_at='fixture',child=dict(pid=group,start_ticks=M._proc(group)[0]))
 lock.write_text(json.dumps(meta));original_path=M.Path;original_owner=os.environ.get('BF16_CONV_GUARD_OWNER')
 M.Path=lambda p:lock if str(p)=='/run/lock/ascend-npu/device-0.lock' else original_path(p)
 os.environ['BF16_CONV_GUARD_OWNER']='conv-cpu-fixture'
 try:
  with lock.open('r+') as held:
   fcntl.flock(held.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   assert M._guard_identity() is not None
   for field,value in [('boot_id','wrong'),('start_ticks',0),('owner','wrong'),('phase','waiting')]:
    changed=dict(meta);changed[field]=value;lock.write_text(json.dumps(changed));assert M._guard_identity() is None
   changed=copy.deepcopy(meta);changed['child']['start_ticks']=0;lock.write_text(json.dumps(changed));assert M._guard_identity() is None
   lock.write_text(json.dumps(meta));fcntl.flock(held.fileno(),fcntl.LOCK_UN)
   assert M._guard_identity() is None
 finally:
  M.Path=original_path
  if original_owner is None:os.environ.pop('BF16_CONV_GUARD_OWNER',None)
  else:os.environ['BF16_CONV_GUARD_OWNER']=original_owner
passed('real CPU fixture flock and live boot/proc/pgrp; stale owner/phase/startticks/boot and unlocked lock rejected')
# Actual admission orchestration, with only capacity/guard/environment boundaries
# stubbed. The original resource gate is separately called on actual CPU cgroups.
actual_capacity=private['resource_gate'].capacity()
assert actual_capacity['effective_available_host_bytes'] >= 0
saved_admission={n:getattr(M,n) for n in ('_require_guard','_environment','_snapshot','_load','_loaded_vendor_pins','_new_run','_device_capacity')}
with tempfile.TemporaryDirectory(dir=O) as td:
 d=Path(td);trace=[];en=M.ENVELOPES[realcase.id]
 host={'effective_available_host_bytes':en['host_required']}
 free=en['device_required']
 M._require_guard=lambda:('identity',)
 M._environment=lambda:('environment',)
 M._snapshot=lambda runtime=False:{}
 M._load=lambda data:{'resource_gate':types.SimpleNamespace(capacity=lambda:dict(host))}
 M._loaded_vendor_pins=lambda require=False:None
 M._new_run=lambda key:d
 def hbm():trace.append('hbm');return free,free
 M._device_capacity=hbm
 try:
  admitted=M._prepare_admission(('ops-transformer',op,realcase.id))
  evidence=json.loads((d/'RESOURCE_ADMISSION.json').read_text())
  assert evidence['stage']=='before_input_allocation'
  assert all(evidence[k]==en[k] for k in ('host_required','device_required','disk_required'))
  trace.clear();host['effective_available_host_bytes']-=1
  reject(lambda:M._prepare_admission(('ops-transformer',op,realcase.id)));assert not trace
  host['effective_available_host_bytes']+=1
  disk_usage=M.shutil.disk_usage
  try:
   M.shutil.disk_usage=lambda path:types.SimpleNamespace(free=en['disk_required']-1)
   reject(lambda:M._prepare_admission(('ops-transformer',op,realcase.id)));assert not trace
  finally:M.shutil.disk_usage=disk_usage
  free-=1
  reject(lambda:M._prepare_admission(('ops-transformer',op,realcase.id)))
 finally:
  for n,v in saved_admission.items():setattr(M,n,v)
passed('actual admission orchestration enforces original host/disk/HBM envelope and writes pre-input receipt; real CPU cgroup accounting read')
# Wrapper state-machine tests use mocks; these assert no numeric/ABI result.
saved={n:getattr(M,n) for n in ('_snapshot','_guard_identity','_require_guard','_environment','_prepare_admission','_loaded_vendor_pins')}
events=[];identity=('guard',1);environment=('env',1);run_dir=O/'mock_run';run_dir.mkdir(exist_ok=True)
def admission(k):
 events.append('admission');return ({'binding':types.SimpleNamespace(dense_fp32=build), 'adapter':types.SimpleNamespace(candidate=lambda c,native_factory:native_factory(c))},run_dir,identity,environment)
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
 # Also adapter post-factory failure and actual call failure own the same cleanup.
 M._loaded_vendor_pins=lambda require=False:None
 original_admission=M._prepare_admission
 def failing_admission(k):
  private,d,i,e=admission(k)
  def fail(c,native_factory):
   native_factory(c);raise RuntimeError('adapter construction failed')
  private['adapter']=types.SimpleNamespace(candidate=fail)
  return private,d,i,e
 M._prepare_admission=failing_admission;cand=M.make(c);events.clear();reject(cand.implementation.run);assert events[-1]=='cleanup'
 M._prepare_admission=original_admission
 passed('cleanup also runs when required map check fails after native construction')
 M._loaded_vendor_pins=lambda require=False:None;M._guard_identity=lambda:None
 events.clear();cand=M.make(c);assert 'admission' not in events;cand.implementation.run();assert events.count('admission')==1
 passed('CPU construction is lazy; run with no preadmission requests fresh admission')
finally:
 for n,v in saved.items():setattr(M,n,v)
# Test actual environment validation without importing providers.
original_env=dict(os.environ);prefix=sys.prefix
try:
 os.environ.update(M.ENV);os.environ.update(LD_LIBRARY_PATH=M.LD_PATH,PYTHONPATH=M.PYTHON_PATH,BF16_CONV_RUN_ROOT='/tmp/conv-registration-environment-test')
 for n in M.CACHE_NAMES:os.environ[n]='/tmp/conv-registration-environment-test/'+n
 sys.prefix='/root/dyq/workspace/.venv';M._environment()
 os.environ['ASCEND_CLASSIC_TRANSFORMER_VENDOR']='/wrong';reject(M._environment)
finally:os.environ.clear();os.environ.update(original_env);sys.prefix=prefix
passed('actual environment validator rejects changed vendor; no device imports')
assert not forbidden
(run_dir/'loaded-maps.txt').unlink(missing_ok=True)
run_dir.rmdir()
# Full real pin check is CPU file hashing, not a mocked runtime validation.
M._snapshot(runtime=True)
passed('all frozen runtime/source/dependency bytes rehashed unchanged')
checker=Path('/root/dyq/workspace/TileOPs/scripts/ascend_pair_check.py').read_text()
assert checker.index('candidate = candidate_factory(case, args.library)()') < checker.index('impl_args = case.inputs')
passed('actual canonical checker constructs Candidate before first Conv inputs access')
result={'passed':True,'checks':checks,'device_import_attempts':forbidden,'npu_runs':0,'numerical_runs':0,'native_library_loads':0,'mock_numeric_claims':False,'actual_host_effective_available_bytes':actual_capacity['effective_available_host_bytes']}
(O/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
