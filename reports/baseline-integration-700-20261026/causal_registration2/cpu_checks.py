"""CPU stub checks only; --published uses fresh actual registry imports."""
import sys
sys.dont_write_bytecode=True
import copy,importlib,importlib.abc,json,types,os
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent
sys.path.insert(0,'/root/dyq/workspace/TileOPs')
class BlockRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','torch_npu','acl','tvm','tilelang'}:raise AssertionError('Forbidden runtime '+fullname)
sys.meta_path.insert(0,BlockRuntime())
import ctypes
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Forbidden CDLL'))
provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer')
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase,Candidate
name='benchmarks.ascend_all_pairs.ops_transformer.integration700_causal_registration2'
published='--published' in sys.argv
if published:m=importlib.import_module(name)
else:
 m=types.ModuleType(name);m.__file__=str(P/'integration700_causal_registration2.py')
 exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__)
 sys.modules[name]=m;setattr(provider,name.rsplit('.',1)[1],m)
checks=[]
def reject(fn,typ=Exception):
 try:fn()
 except typ:return
 raise AssertionError('Expected rejection')
def case(key,params=None):
 return types.SimpleNamespace(id=key[2],params=json.loads(m.CASE_PARAMS[key]) if params is None else params,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])))
prior=dict(provider.PROVIDERS)
if not published:exec(compile((P/'registration.append.py').read_bytes(),'registration.append.py','exec'),provider.__dict__)
for key in m.CASE_PARAMS:
 c=case(key);assert m.matches(c)
 candidate=candidate_factory(c,key[0])();assert isinstance(candidate,Candidate)
 assert candidate.api=='ops-transformer::aclnnCausalConv1dUpdate'
 reject(lambda:m.make(c,'ops-nn'),UnsupportedCase)
 for param in c.params:
  p=copy.deepcopy(c.params);p.pop(param);reject(lambda:candidate_factory(case(key,p),key[0])(),UnsupportedCase)
  p=copy.deepcopy(c.params);p[param]={'changed':True};reject(lambda:m.make(case(key,p)),UnsupportedCase)
 p=copy.deepcopy(c.params);p['extra']=0;reject(lambda:m.make(case(key,p)),UnsupportedCase)
 with patch.object(Path,'read_text',return_value='{}'):reject(lambda:candidate.implementation.run(),RuntimeError)
checks.append('actual registry 2/2 exact Candidate routes; all parameter deletion/change and extra parameter rejected; wrong baseline rejected; no unguarded build')
for op in ('CausalConv1dUpdateFwdOp',):
 route=provider.PROVIDERS[op];assert route.__module__==m.__name__
 if not published:assert route.previous is prior[op]
 sentinel=object();seen=[];wrapped=m.wrap(lambda c:(seen.append(c.id) or sentinel))
 c=types.SimpleNamespace(id='unknown',params={},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 assert wrapped(c) is sentinel and seen==['unknown']
 # Actual published wrapper must retain the previous factory closure.
 previous=route.__closure__[0].cell_contents
 assert previous is route.previous and callable(previous)
 cell=route.__closure__[0]
 cell.cell_contents=lambda c:sentinel
 try:assert candidate_factory(c,'ops-transformer')() is sentinel
 finally:cell.cell_contents=previous
 reject(lambda:m.wrap(None)(c),UnsupportedCase)
if not published:
 for op,f in prior.items():
  if op not in ('CausalConv1dUpdateFwdOp',):assert provider.PROVIDERS[op] is f
checks.append('previous factory preserved for unknown cases; unrelated provider identities preserved in staging')
data=m._snapshot(); original=Path.read_bytes
for path in data:
 with patch.object(Path,'read_bytes',lambda p,target=path:original(p)+(b'bad' if str(p)==target else b'')):
  reject(m._snapshot,RuntimeError)
for path,digest in m.PINS.items():
 with patch.object(Path,'read_bytes',return_value=b'simulated corrupt dependency'):
  reject(lambda:m._checked(path,digest),RuntimeError)
checks.append('all pinned dependency hash corruptions rejected via CPU read stubs; local snapshot corruption rejected without disk mutation')
poison=types.ModuleType('direct_acl')
with patch.dict(sys.modules,{'direct_acl':poison,'binding':poison}):
 private=m._load(data)
 assert private['binding'].DirectACL is private['direct_acl'].DirectACL
 assert private['binding'].require_guard is private['direct_acl'].require_guard
 assert sys.modules['direct_acl'] is poison and sys.modules['binding'] is poison
 assert private['binding'].__file__==str(m.ROOT/'binding.py')
 for path,raw in data.items():
  with patch.object(Path,'read_bytes',side_effect=AssertionError('mutable read')),patch.object(Path,'read_text',side_effect=AssertionError('mutable read')):
   assert private['binding'].Path(path).read_bytes()==raw
   assert private['binding'].Path(path).read_text()==raw.decode()
checks.append('private direct_acl/binding immune to sys.modules poison; all five local behavior reads frozen; original __file__ preserved')
# Exercise the unchanged guard with CPU metadata stubs, including exact pgrp/owner.
with patch.dict(os.environ,{'GDN_CASE_OWNER':'cpu-only-owner'}):
 private=m._load(data)
 metadata={'owner':'cpu-only-owner','phase':'running','child':{'pid':os.getpgrp()}}
 with patch.object(Path,'read_text',return_value=json.dumps(metadata)):private['direct_acl'].require_guard()
 for bad in ({**metadata,'owner':'wrong'},{**metadata,'phase':'waiting'},{**metadata,'child':{'pid':os.getpgrp()+1}}):
  with patch.object(Path,'read_text',return_value=json.dumps(bad)):reject(private['direct_acl'].require_guard,RuntimeError)
checks.append('unchanged original guard rejects owner/phase/foreground processgroup mismatch')
# Synthetic loaded mappings; no aarch64 library load or file writes.
lines=[]
for p in m.LOADED_VENDOR:lines.append('0000-0001 r--p 00000000 00:01 123 '+p)
fake_stat=types.SimpleNamespace(st_ino=123,st_dev=os.makedev(0,1))
with patch.object(Path,'read_text',return_value='\n'.join(lines)),patch.object(Path,'stat',return_value=fake_stat),patch.object(m,'_checked',return_value=b''):
 m._loaded_vendor_pins(require=True)
 with patch.object(Path,'read_text',return_value=lines[0]+' (deleted)'):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=lines[0].replace(' 123 ',' 124 ')):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=lines[0].replace('/install/','/wrong/')):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=''):reject(lambda:m._loaded_vendor_pins(require=True),RuntimeError)
checks.append('synthetic loaded-vendor path/inode/deleted/missing mapping rejections; mapped identity checked before hash')
# Wrapper lifecycle with a stub child: no numerical or native execution.
import tempfile
with tempfile.TemporaryDirectory(dir=P,prefix='cpu-stub-') as runroot:
 calls=[]; cleaned=[]; mapping=[]
 proxy=types.SimpleNamespace(environ={})
 child=Candidate(types.SimpleNamespace(run=lambda *a,**k:(calls.append((a,k)) or ('y','full_state4'))),'stub','stub',{},cleanup=lambda:cleaned.append(True))
 binding=types.SimpleNamespace(os=proxy,build=lambda c:(calls.append(c.id) or child))
 priv={'binding':binding}
 candidate=m.make(case(next(iter(m.CASE_PARAMS))))
 assert calls==[]
 with patch.object(m,'_load',return_value=priv),patch.object(m,'_snapshot',return_value={}),patch.object(m,'_require_runtime',return_value=None),patch.object(m,'_loaded_vendor_pins',side_effect=lambda require=False:mapping.append(require)),patch.dict(os.environ,{'GDN_RUN_DIR':runroot}):
  assert candidate.implementation.run('input')==('y','full_state4')
  first=proxy.environ['GDN_RUN_DIR']
  assert candidate.implementation.run('next')==('y','full_state4')
  second=proxy.environ['GDN_RUN_DIR']; assert first!=second
  assert Path(first).is_dir() and Path(second).is_dir()
  candidate.cleanup(); assert cleaned==[True]
 assert sum(isinstance(x,str) for x in calls)==1
 assert mapping==[False,False,True,True,False,True]
checks.append('stub lazy child built once; arguments/results/cleanup forwarded; host_roundtrip; independent invocation evidence directories; pre/post mapping checks')
assert not any(x in sys.modules for x in ('torch','torch_npu','acl','tvm','tilelang'))
result=dict(status='passed',mode='published fresh registry' if published else 'private in-process snippet staging',checks=checks,real_runtime_imports=0,npu_runs=0,numerical_validation=False,accepted_added=0)
print(json.dumps(result,indent=2))
if not published:(P/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
