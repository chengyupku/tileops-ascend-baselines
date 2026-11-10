import builtins,hashlib,json,os,sys,types,mmap,tempfile
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,'/root/dyq/workspace/TileOPs')
original_import=builtins.__import__; forbidden=[]
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl'):
  forbidden.append(name);raise AssertionError('Forbidden device import '+name)
 return original_import(name,*a,**kw)
builtins.__import__=protected
M=types.ModuleType('registration');M.__file__=str(O/'integration700_bf16_gemm_registration18.py')
exec(compile(Path(M.__file__).read_bytes(),M.__file__,'exec'),M.__dict__)
checks=[]
def passed(x):checks.append(x)
def reject(fn):
 try:fn()
 except (RuntimeError,ValueError,AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
def case(k,p):return types.SimpleNamespace(id=k[2],params=p,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=k[1])))
cs=[case(k,json.loads(v)) for k,v in M.CASE_PARAMS.items()]
for c in cs:
 assert M.matches(c); candidate=M.make(c); assert callable(candidate.implementation.run)
 assert not M.matches(c,'wrong-baseline')
 bad=case(('ops-transformer','GemmFwdOp',c.id),dict(c.params,unexpected=True));assert not M.matches(bad);reject(lambda:M.make(bad))
passed('18 real canonical params CPU construction; wrong baseline/full params rejected')
previous=[]
def old(c):previous.append(c);return 'old'
f=M.wrap(old)
for c in cs:
 unknown=case(('ops-transformer','GemmFwdOp','old-'+c.id),c.params);assert f(unknown)=='old'
assert len(previous)==18;passed('18 previous-route mockfactory calls preserved; unknown keys delegated')
# Native dependencies must be private even when all names are poisoned.
poison={n:sys.modules.get(n) for n in ('binding','resource_gate','benchmarks.ascend_math_more','benchmarks.ascend_baselines')}
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
events=[];identity=('guard',1);environment=('env',1);run_dir=O/'mock_run';run_dir.mkdir(exist_ok=True)
def admission(k):
 events.append('admission');return ({'binding':types.SimpleNamespace(dense_fp32=build)},run_dir,identity,environment)
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
 M._loaded_vendor_pins=lambda require=False:None;M._guard_identity=lambda:None
 events.clear();cand=M.make(c);assert 'admission' not in events;cand.implementation.run();assert events.count('admission')==1
 passed('CPU construction is lazy; run with no preadmission requests fresh admission')
finally:
 for n,v in saved.items():setattr(M,n,v)
# Test actual environment validation without importing providers.
original_env=dict(os.environ);prefix=sys.prefix
try:
 os.environ.update(M.ENV);os.environ.update(LD_LIBRARY_PATH=M.LD_PATH,PYTHONPATH=M.PYTHON_PATH,BF16_GEMM_RUN_ROOT=str(O/'environment-test'))
 for n in M.CACHE_NAMES:os.environ[n]=str(O/'environment-test'/n)
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
passed('actual canonical checker constructs Candidate before first Gemm inputs access')
result={'passed':True,'checks':checks,'device_import_attempts':forbidden,'npu_runs':0,'numerical_runs':0,'native_library_loads':0,'mock_numeric_claims':False}
(O/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
