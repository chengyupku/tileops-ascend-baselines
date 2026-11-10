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
name='benchmarks.ascend_all_pairs.ops_transformer.integration700_attention_registration5'
published='--published' in sys.argv
if published:m=importlib.import_module(name)
else:
 m=types.ModuleType(name);m.__file__=str(P/'integration700_attention_registration5.py')
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
 assert candidate.api=='ops-transformer::aclnnFlashAttnMetadata + aclnnFlashAttn'
 reject(lambda:m.make(c,'ops-nn'),UnsupportedCase)
 for param in c.params:
  p=copy.deepcopy(c.params);p.pop(param);reject(lambda:candidate_factory(case(key,p),key[0])(),UnsupportedCase)
  p=copy.deepcopy(c.params);p[param]={'changed':True};reject(lambda:m.make(case(key,p)),UnsupportedCase)
 p=copy.deepcopy(c.params);p['extra']=0;reject(lambda:m.make(case(key,p)),UnsupportedCase)
 with patch.object(Path,'read_text',return_value='{}'):reject(lambda:candidate.implementation.run(),RuntimeError)
checks.append('actual registry 5/5 exact Candidate routes; all parameter deletion/change and extra parameter rejected; wrong baseline rejected; no unguarded build')
for op in ('GQAPagedFwdOp','GQAVarlenFwdOp'):
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
  if op not in ('GQAPagedFwdOp','GQAVarlenFwdOp'):assert provider.PROVIDERS[op] is f
checks.append('previous factory preserved for unknown cases; unrelated provider identities preserved in staging')
data=m._snapshot(); original=Path.read_bytes
for path in data:
 with patch.object(Path,'read_bytes',lambda p,target=path:original(p)+(b'bad' if str(p)==target else b'')):
  reject(m._snapshot,RuntimeError)
for path,digest in m.PINS.items():
 with patch.object(Path,'read_bytes',return_value=b'simulated corrupt dependency'):
  reject(lambda:m._checked(path,digest),RuntimeError)
checks.append('all 60 dependency hash corruptions rejected via CPU read stubs; local snapshot corruption rejected without disk mutation')
poison=types.ModuleType('direct_api')
with patch.dict(sys.modules,{'direct_api':poison,'firstcase_adapter':poison,'remaining_adapter':poison}):
 private=m._load(data)
 for n in ('firstcase_adapter','remaining_adapter'):
  load=private[n].__dict__['__builtins__']['__import__']
  assert load('direct_api',fromlist=['flash_attn']) is private['direct_api']
 assert sys.modules['direct_api'] is poison
 assert private['direct_api'].__file__==str(m.ROOT/'direct_api.py')
checks.append('module-cache poisoning cannot replace private verified direct_api; shared sys.modules untouched; original __file__ preserved')
# Stub source read: private Path returns exact frozen bytes, not the mutable filesystem.
private=m._load(dict(data,**{m.SOURCE:b'frozen helper source'}))
assert private['direct_api'].Path(m.SOURCE).read_text()=='frozen helper source'
checks.append('AST allocation-helper source read bound to verified bytes')
# Synthetic loaded mappings; no aarch64 library load or file writes.
lines=[]
for p in m.LOADED_VENDOR:lines.append('0000-0001 r--p 00000000 00:01 123 '+p)
fake_stat=types.SimpleNamespace(st_ino=123,st_dev=os.makedev(0,1))
with patch.object(Path,'read_text',return_value='\n'.join(lines)),patch.object(Path,'stat',return_value=fake_stat),patch.object(m,'_checked',return_value=b''):
 m._loaded_vendor_pins(require=True)
 with patch.object(Path,'read_text',return_value=lines[0]+' (deleted)'):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=lines[0].replace(' 123 ',' 124 ')):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=lines[0].replace('/native_install/','/wrong/')):reject(m._loaded_vendor_pins,RuntimeError)
 with patch.object(Path,'read_text',return_value=''):reject(lambda:m._loaded_vendor_pins(require=True),RuntimeError)
checks.append('synthetic loaded-vendor path/inode/deleted/missing mapping rejections; mapped identity checked before hash')
assert not any(x in sys.modules for x in ('torch','torch_npu','acl','tvm','tilelang'))
result=dict(status='passed',mode='published fresh registry' if published else 'private in-process snippet staging',checks=checks,real_runtime_imports=0,npu_runs=0,numerical_validation=False,accepted_added=0)
print(json.dumps(result,indent=2))
if not published:(P/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
