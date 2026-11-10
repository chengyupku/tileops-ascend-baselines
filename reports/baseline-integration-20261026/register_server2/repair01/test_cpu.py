"""No real torch/ACL imports. Exercise real registry/factories and source bodies."""
import ast,copy,hashlib,importlib,importlib.abc,json,os,sys,types
from pathlib import Path
from unittest.mock import patch
W=Path('/opt/tileops-all-pairs/workspace');P=W/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer';D=W/'reports/baseline-integration-20261026/register_server2/repair01'
sys.dont_write_bytecode=True;sys.path.insert(0,str(W/'TileOPs'))
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','torch_npu','acl','tensorflow','numpy'):raise AssertionError('Forbidden real runtime import: '+fullname)
sys.meta_path.insert(0,Deny())
checks=[]
def ok(name):checks.append(name)
def rejects(name,fn):
 try:fn()
 except (RuntimeError,ValueError,FileNotFoundError,KeyError):ok(name);return
 raise AssertionError('Not rejected: '+name)
class Tensor:
 def __init__(self,shape,dtype='bfloat16',device='npu:0'):self.shape=tuple(shape);self.dtype=dtype;self.device=device
 def contiguous(self):return self
 def clone(self):return Tensor(self.shape,self.dtype)
 def float(self):return Tensor(self.shape,'float32')
 def transpose(self,a,b):s=list(self.shape);s[a],s[b]=s[b],s[a];return Tensor(s,self.dtype)
 def reshape(self,*s):
  if len(s)==1 and isinstance(s[0],(tuple,list)):s=s[0]
  s=list(s)
  if -1 in s:
   import math
   s[s.index(-1)]=int(math.prod(self.shape)/-math.prod(s))
  return Tensor(s,self.dtype)
class Candidate:
 def __init__(self,implementation,*args,**kwargs):self.implementation=implementation;self.kwargs=kwargs
class Implementation:
 def __init__(self,run):self.run=run
base=types.ModuleType('benchmarks.ascend_baselines');base.Candidate=Candidate;base.UnsupportedCase=type('UnsupportedCase',(RuntimeError,),{});base.LIBRARIES=W/'ascend_baselines';base.WORKSPACE=W;base.source_hashes=lambda *a:{};base.local_function=lambda *a:None
sys.modules[base.__name__]=base
api=types.ModuleType('benchmarks.api');api.Implementation=Implementation;sys.modules[api.__name__]=api
classic=types.ModuleType('benchmarks.ops.bench_ascend_classics');classic.candidate_factories=lambda c:{'ops-transformer':lambda:('historical',c.id)};sys.modules[classic.__name__]=classic
registry=importlib.import_module('benchmarks.ascend_all_pairs.registry');provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');v=importlib.import_module(provider.__name__+'.validated_runtime')
assert provider.PROVIDERS['BmmFwdOp'].__module__.endswith('.adapters');ok('original BMM route preserved')
for group in ['attention','gdn']:v.verify_pins(group);ok(group+' live source/runtime pins and required directories')
# AST identity of every non-import top-level node, including complete functions/classes.
for name in v.MANIFEST['sources']:
 f,relocated,_=v.relocated_source(name)
 clean=lambda s:[ast.dump(n,include_attributes=False) for n in ast.parse(s).body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 assert clean(Path(f).read_text())==clean(relocated)
ok('all original function/class AST identical; import relocation only')
for group in ['attention','gdn']:
 cfg=v.MANIFEST['groups'][group];orig=cfg['pins'];dirs=cfg['directories']
 with patch.dict(cfg,{'pins':{'/nonexistent/source.py':'0'},'directories':[]}):rejects(group+' missing source',lambda:v.verify_pins(group))
 with patch.dict(cfg,{'pins':{},'directories':['/nonexistent/vendor']}):rejects(group+' missing vendor',lambda:v.verify_pins(group))
 f=next(iter(orig))
 with patch.dict(cfg,{'pins':{f:'0'},'directories':[]}):rejects(group+' wrong source hash',lambda:v.verify_pins(group))
 rejects(group+' environment mismatch before import',lambda:v.check_environment(group))
# Mock only tensor allocation and native transport; original factories and run bodies execute.
torch=types.ModuleType('torch');torch.float32='float32';torch.int32='int32';torch.bfloat16='bfloat16'
torch.empty=lambda s,device=None,dtype=None:Tensor(s,dtype,device)
torch.empty_like=lambda t:Tensor(t.shape,t.dtype,t.device)
torch.zeros=torch.empty;torch.full=lambda s,value,**kw:Tensor(s,kw['dtype'],kw['device'])
sys.modules['torch']=torch
sys.modules['binding']=types.SimpleNamespace(build=lambda c:(_ for _ in ()).throw(AssertionError('global binding collision')))
sys.modules['direct_acl']=types.SimpleNamespace(DirectACL=None)
direct=v.load('direct_acl');direct.require_guard=lambda:None
calls=[]
class Transport:
 def __init__(self,*args):self.torch=torch;self.vendor=Path(v.MANIFEST['groups']['attention']['vendor'])
 def cleanup(self):pass
 def call(self,*args):calls.append(args)
direct.DirectACL=Transport
v.load('ascend_graph_call').GraphCall=Transport
resource=v.load('resource_gate');resource_calls=[]
resource.require_capacity=lambda here,run:resource_calls.append((str(here),str(run)))
def make(cid):
 c=v.MANIFEST['contracts'][cid];s=c['snapshot'];inputs=[]
 for n in s['declared_input_names']:
  d=s['declared_tensors'].get(n);inputs.append(Tensor(*d) if d else None)
 return types.SimpleNamespace(id=cid,params=copy.deepcopy(s['parameters']),inputs=inputs,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=c['op'])))
with patch.object(v,'verify_pins'),patch.object(v,'require_guard'),patch.object(v,'check_environment',side_effect=lambda g:Path(v.MANIFEST['run_root'])/current[0]):
 current=['']
 for cid,c in v.MANIFEST['contracts'].items():
  current[0]=cid;case=make(cid);factory=registry.candidate_factory(case,'ops-transformer');result=factory();assert isinstance(result,Candidate)
  # Varlen offset tensors require actual boundary content; factory still checked for all 39.
  if case.params.get('cu_seqlens') is None:
   result_value=result.implementation.run(*case.inputs)
   if c['group']=='gdn':assert len(result_value)==2;assert result_value[1].shape==tuple(case.params['final_state'][0])
   else:
    attrs=calls[-1][-1];assert attrs['input_layout']=='BSND' and attrs['keep_prob']==1.0 and attrs['head_num']==case.inputs[0].shape[2]
   ok('registry + original run with mock transport '+cid)
  else:ok('registry + original factory '+cid)
  changed=make(cid);changed.params['unexpected_contract_change']=True
  rejects('contract mutation '+cid,lambda:registry.candidate_factory(changed,'ops-transformer')())
  altered=make(cid);altered.inputs[0].shape=(1,)
  rejects('input mutation '+cid,lambda:registry.candidate_factory(altered,'ops-transformer')())
assert len(resource_calls)==26;ok('GDN original resource gate invoked for every accepted key')
for op in ['GDNFwdOp','GQADenseFwdOp','UnknownOp','DeltaNetFwdOp']:
 case=types.SimpleNamespace(id='unverified',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert registry.candidate_factory(case,'ops-transformer')()==('historical','unverified')
for row in json.loads((D.parent/'UNINTEGRATED_KEYS.json').read_text()):
 _,op,cid=row['key'];assert cid not in v.MANIFEST['contracts']
 case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert registry.candidate_factory(case,'ops-transformer')()==('historical',cid)
ok('all 10 excluded keys preserve historical route; none accepted')
ok('unknown and unverified routes use actual historical candidate_factories API')
assert v.load('binding') is not sys.modules['binding'];ok('global binding/direct_acl cache collision isolated')
key=v.NAMESPACE+'.binding';saved=sys.modules[key];sys.modules[key]=types.ModuleType(key)
rejects('private namespace poisoned cache rejected',lambda:v.load('binding'));sys.modules[key]=saved
# Startup-only LD validation, including mutation after process startup.
with patch.dict(os.environ,v.MANIFEST['groups']['gdn']['environment']):rejects('post-import LD mutation cannot impersonate exec environment',lambda:v.check_environment('gdn'))
# CPU launcher builds exact argv/env, does not execute it.
sys.path.insert(0,str(P));launch=importlib.import_module('launch_verified')
# launcher references same implementation via standalone module; patch its guards only.
lv=sys.modules['validated_runtime']
for group in ['attention','gdn']:
 cid=next(k for k,c in v.MANIFEST['contracts'].items() if c['group']==group)
 run=Path(v.MANIFEST['run_root'])/'cpu-plan-test'/cid
 with patch.object(launch,'verify_pins'),patch.object(launch,'require_guard'),patch.object(launch,'load',return_value=resource),patch.dict(sys.modules,{'torch':None}):
  # Remove mock runtime entirely for the pure CPU launcher.
  savedtorch=sys.modules.pop('torch')
  try:
   with patch.object(Path,'mkdir'):
    argv,env,cwd=launch.prepare(group,cid,run,['/caller.py','--x'])
  finally:sys.modules['torch']=savedtorch
 assert argv[-2:]==['/caller.py','--x']
 cfg=v.MANIFEST['groups'][group]
 assert all(env[k]==value for k,value in cfg['environment'].items())
 if group=='gdn':assert env['LD_LIBRARY_PATH'].split(':')[:2]==cfg['environment']['LD_LIBRARY_PATH'].split(':')[:2];assert cwd==Path(cfg['cwd'])
 ok(group+' launch plan exact environment/order/cwd and ordinary caller argv')
 if group=='gdn':assert resource_calls[-1][1]==str(run);ok('resource gate before exec')
 with patch.object(launch,'prepare',return_value=(argv,env,cwd)),patch.object(launch.os,'chdir') as cd,patch.object(launch.os,'execve') as ex,patch.object(sys,'argv',['launch_verified.py',group,cid,str(run),'--','/caller.py','--x']):
  launch.main();cd.assert_called_once_with(cwd);ex.assert_called_once_with(argv[0],argv,env)
 ok(group+' execve ordinary caller tested without execution')
(D/'STATIC_CHECKS.json').write_text(json.dumps({'status':'passed','checks':checks,'count':len(checks),'real_runtime_imports':0,'NPU_operations':0,'numerical_replays':0,'mock_boundary':'tensor allocation/native transport, guard/resource observation; full original functions and actual registry/factories executed'},indent=2))
print('PASS',len(checks))
