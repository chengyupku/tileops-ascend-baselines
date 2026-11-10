import sys,builtins,types,dataclasses,copy,json,datetime,ctypes,tempfile,shutil
from pathlib import Path
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O));blocked=[];dl=[];orig=builtins.__import__
def protected(name,*a,**kw):
 if name.split('.')[0] in ('torch','torch_npu','acl','tilelang','tvm'):
  blocked.append(name);raise AssertionError('device import '+name)
 return orig(name,*a,**kw)
builtins.__import__=protected
def audit(event,args):
 if event=='ctypes.dlopen':dl.append(str(args));raise AssertionError('dlopen forbidden')
sys.addaudithook(audit)
import build_contracts as B
import registration as M
from benchmarks.ascend_baselines import UnsupportedCase,Candidate
from benchmarks.ascend_all_pairs.ops_transformer import PROVIDERS
checks=[];cs=B.cases();data=M.snapshot();before=dict(PROVIDERS);private=dict(PROVIDERS);M.install(private)
def rejects(f,exc=UnsupportedCase):
 try:f()
 except exc as e:
  if exc is UnsupportedCase:assert type(e) is UnsupportedCase
  return
 raise AssertionError('Expected '+exc.__name__)
def args(c):return [None if c._call.specs.get(n) is None else types.SimpleNamespace(shape=c._call.specs[n].shape,dtype=c._call.specs[n].dtype,device='cpu' if c._call.specs[n].cpu else 'npu:0') for n in c._call.signature.inputs]
for c in cs:
 assert M.matches(c)
 candidate=private['DeltaNetFwdOp'](c);assert isinstance(candidate,Candidate);candidate.cleanup()
assert PROVIDERS==before
checks.append('3 true manifest Calls + original passive Case through actual provider dictionary COPY; zero shared writes and lazy construct/cleanup')
mutations=0
for c in cs:
 def bad(call):
  global mutations
  rejects(lambda:private['DeltaNetFwdOp'](B.Case(c.id,call,None)));mutations+=1
 for k in c._call.params:bad(dataclasses.replace(c._call,params=dict(c._call.params,**{k:'wrong'})))
 bad(dataclasses.replace(c._call,params=dict(c._call.params,out=True)))
 for k in c._call.indices:bad(dataclasses.replace(c._call,indices=dict(c._call.indices,**{k:'wrong'})))
 for n,s in c._call.specs.items():
  if s is None:continue
  for fields in ({'shape':(1,)},{'dtype':'float64'},{'cpu':not s.cpu},{'values':[999]}):
   specs=dict(c._call.specs);specs[n]=dataclasses.replace(s,**fields);bad(dataclasses.replace(c._call,specs=specs))
  specs=dict(c._call.specs);specs[n]=None;bad(dataclasses.replace(c._call,specs=specs))
 for n in ('initial_state','cu_seqlens','cu_seqlens_cpu','out'):
  specs=dict(c._call.specs);specs[n]=c._call.specs['q'];bad(dataclasses.replace(c._call,specs=specs))
 cand=M.make(c);rejects(lambda:cand.implementation.run(*args(c)),RuntimeError);cand.cleanup()
 for i,value in enumerate(args(c)):
  a=args(c);a[i]=object() if value is None else types.SimpleNamespace(shape=(1,),dtype=value.dtype,device=value.device)
  rejects(lambda:M.validate_inputs(c,a,{}))
 rejects(lambda:M.validate_inputs(c,args(c),{'out':object()}))
 seen=[];f=M.wrap(lambda x:seen.append(x) or 'previous');unknown=types.SimpleNamespace(id='unknown',_call=c._call)
 assert f(unknown)=='previous' and seen==[unknown]
checks.append(str(mutations)+' wrong same-name contracts reject public UnsupportedCase; all optional/state/out/offsets fields included; 3 unknown keys previous fallback')
checks.append('3 unguarded actual run attempts fail before device imports; input arity/shape/dtype/device/optional rejection')
saved={k:getattr(M,k) for k in ('admit','environment','require_guard','private_build','loaded_libraries')}
try:
 identity=['first'];admissions=[];counts={'build':0,'run':0,'cleanup':0}
 M.require_guard=lambda:identity[0];M.environment=lambda pin:'env';M.loaded_libraries=lambda pin:None
 def admit(pin,full):admissions.append(full);return identity[0],'env'
 M.admit=admit
 def build(c,p,d):
  counts['build']+=1
  def run(*a):counts['run']+=1;return 'fixture'
  def clean():counts['cleanup']+=1
  return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=clean)
 M.private_build=build;c=next(c for c in cs if c.id=='prefill-12k-continued-bfloat16');candidate=M.make(c)
 assert counts['build']==0
 assert candidate.implementation.run(*args(c))=='fixture';candidate.implementation.run(*args(c))
 assert admissions==[True,False] and counts['build']==1
 identity[0]='changed';rejects(lambda:candidate.implementation.run(*args(c)),RuntimeError);rejects(candidate.cleanup,RuntimeError)
 assert counts['run']==2 and counts['cleanup']==0
 identity[0]='first';candidate.cleanup();candidate.cleanup();assert counts['cleanup']==1
 rejects(lambda:candidate.implementation.run(*args(c)),RuntimeError)
 checks.append('guard lifetime reuse/cleanup fail closed; full dependency validation once per candidate; exactly-once cleanup')
 for failrun,failclean in [(True,False),(False,True),(True,True)]:
  calls=[]
  def build(c,p,d):
   def run(*a):
    calls.append('run')
    if failrun:raise RuntimeError('fixture run')
   def clean():
    calls.append('clean')
    if failclean:raise RuntimeError('fixture cleanup')
   return types.SimpleNamespace(implementation=types.SimpleNamespace(run=run),cleanup=clean)
  M.private_build=build;candidate=M.make(c)
  if failrun:rejects(lambda:candidate.implementation.run(*args(c)),RuntimeError)
  else:
   candidate.implementation.run(*args(c));rejects(candidate.cleanup,RuntimeError)
  candidate.cleanup();assert calls==['run','clean']
 checks.append('run and cleanup failure fixtures close exactly once without suppressed cleanup errors')
finally:
 for k,v in saved.items():setattr(M,k,v)
# Verify real contract parameters agree with actual checked receipt, including presence and returned state dtype.
source=O.parent
for row in json.loads((O/'AUDIT.json').read_text()):
 c=next(c for c in cs if c.id==row['exactkey'][2]);receipt=json.loads((source/row['checker']).read_text())
 assert M.plain(c.params)==receipt['case_snapshot']['parameters']
 assert list(c._call.signature.inputs)==receipt['case_snapshot']['declared_input_names']
 for n,val in zip(c._call.signature.inputs,receipt['observed_calls']['candidate'][0]['inputs']['items']):
  spec=c._call.specs.get(n)
  assert (spec is None)==(val['kind']=='none'),(c.id,n)
 assert receipt['observed_calls']['candidate'][0]['outputs']['items'][1]['dtype']=='float32'
checks.append('all actual receipt parameters/input presence match fresh manifest; every returned final_state FP32')
# Load complete original modules without invoking native construction, even with poisoned ambient names.
old=sys.modules.get('direct_acl');sys.modules['direct_acl']=types.ModuleType('poison')
try:
 for pin in M.metadata()['RUNTIME_PINS.json'].values():
  mod=M.private_load(pin,data)
  assert callable(mod.build) and mod.require_guard is M.require_guard
  assert mod.DirectACL.__module__.startswith('_linear3_')
 assert sys.modules['direct_acl'].__name__=='poison'
finally:
 if old is None:sys.modules.pop('direct_acl',None)
 else:sys.modules['direct_acl']=old
checks.append('3 actual private original binding/direct_acl module loads are device-free; poisoned ambient direct_acl bypassed')
oldroot=M.DATA_ROOT
with tempfile.TemporaryDirectory(dir=O) as td:
 for n in M.DATA_HASHES:shutil.copyfile(O/n,Path(td)/n)
 M.DATA_ROOT=Path(td)
 try:
  for n in M.DATA_HASHES:
   f=Path(td)/n;raw=f.read_bytes();f.write_bytes(raw+b' ');rejects(M.metadata,RuntimeError);f.write_bytes(raw)
 finally:M.DATA_ROOT=oldroot
checks.append('all contract/local/runtime pin metadata reject hash tampering')
# Verify publication from a copied complete private deployment directory, using the actual provider dictionary copy.
import importlib.util
with tempfile.TemporaryDirectory(dir=O) as td:
 deployed=Path(td)/'published';deployed.mkdir()
 for n in ('registration.py','CONTRACTS.json','LOCAL_PINS.json','RUNTIME_PINS.json'):
  shutil.copyfile(O/n,deployed/n)
 shutil.copytree(O/'closure',deployed/'closure')
 spec=importlib.util.spec_from_file_location('private_linear3_deployed',deployed/'registration.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 assert module.DATA_ROOT==deployed
 providers=dict(PROVIDERS);module.install(providers)
 for c in cs:
  cand=providers['DeltaNetFwdOp'](c);assert isinstance(cand,Candidate);cand.cleanup()
 assert PROVIDERS==before
checks.append('Complete copied deployment closure constructs all 3 real Cases through actual provider COPY without device imports')
ragged=next(c for c in cs if c.id=='varlen-ragged-bfloat16')
assert ragged._call.specs.get('cu_seqlens_cpu') is None
assert 'varlen-4x1k-bfloat16' not in (O/'registration.py').read_text()
checks.append('Ragged has no cu_seqlens_cpu; no equal4x1k special-value check retained')

assert not blocked and not dl
result={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'exact_keys':3,'mutations':mutations,'device_import_attempts':blocked,'dlopen_attempts':dl,'npu_runs':0,'numerical_runs':0,'accepted_added':0,'shared_provider_writes':0}
(O/'CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
