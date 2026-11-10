"""CPU-only isolated tests. --published checks the actual post-publish fresh import."""
import sys
sys.dont_write_bytecode = True
import copy, importlib, importlib.abc, json, os, types
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent; W=Path('/root/dyq/workspace')
sys.path.insert(0,str(W/'TileOPs'))
class BlockRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','torch_npu','acl','tvm','tilelang'}: raise AssertionError('Forbidden real runtime import: '+fullname)
sys.meta_path.insert(0,BlockRuntime())
import ctypes
ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Forbidden CDLL'))
name='benchmarks.ascend_all_pairs.ops_math.integration700_nn_dequant_registration1'
provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_math')
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_baselines import UnsupportedCase
published='--published' in sys.argv
if published:m=importlib.import_module(name)
else:
 m=types.ModuleType(name);m.__file__=str(P/'integration700_nn_dequant_registration1.py');exec(compile(Path(m.__file__).read_bytes(),m.__file__,'exec'),m.__dict__)
 sys.modules[name]=m;setattr(provider,name.rsplit('.',1)[1],m)
def case(params=None, cid=None, op=None):
 return types.SimpleNamespace(params=copy.deepcopy(m.PARAMS if params is None else params),id=cid or m.KEY[2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op or m.KEY[1])))
checks=[]
def reject(fn, typ=Exception):
 try:fn()
 except typ:return
 raise AssertionError('Expected rejection')
assert m.matches(case());assert not m.matches(case(), 'ops-nn')
reject(lambda:m.make(case(), 'ops-nn'),UnsupportedCase)
reject(lambda:m.make(case(op='INT8DequantPerBlockFwdOp')),UnsupportedCase)
reject(lambda:m.make(case(cid='unreviewed')),UnsupportedCase)
for key in m.PARAMS:
 p=copy.deepcopy(m.PARAMS);p.pop(key);reject(lambda:m.make(case(p)),UnsupportedCase)
 p=copy.deepcopy(m.PARAMS);p[key]=None;reject(lambda:m.make(case(p)),UnsupportedCase)
p=copy.deepcopy(m.PARAMS);p['extra']=1;reject(lambda:m.make(case(p)),UnsupportedCase)
p=copy.deepcopy(m.PARAMS);p['q'][0][0]=8192.0;reject(lambda:m.make(case(p)),UnsupportedCase)
checks.append('exact baseline/op/case and every canonical parameter; missing/extra/wrong/type rejected')
pins,data=m.verify_pins();assert len(pins)==57
checks.append('57 current dependency byte hashes match reviewed pins')
original=Path.read_bytes
for target in [str(P/'PINS.json'),*pins]:
 def bad_read(path, target=target):
  raw=original(path);return raw+b'\n# simulated corruption' if str(path)==target else raw
 with patch.object(Path,'read_bytes',bad_read):reject(m.verify_pins,RuntimeError)
checks.append('manifest and each of 57 dependency corruptions rejected via read stub; no shared file mutation')
# Generic module cache poison cannot replace the private verified Python code.
poison=types.ModuleType('poison');poison.make=lambda *a: (_ for _ in ()).throw(AssertionError('poison executed'))
with patch.dict(sys.modules,{m.PREFIX+'.graph':poison,m.PREFIX+'.specs':poison,'adapter':poison}):
 private=m._load(pins,data)
 assert private['graph'] is not poison and private['specs'] is not poison
 assert private['graph'].make.__globals__['SPECS']['MulFwdOp']['graph']=='Mul'
 assert private['adapter'].make_dequant.__globals__['__builtins__']['__import__'](m.PREFIX+'.graph',fromlist=['make']) is private['graph']
checks.append('generic graph/specs/adapter module-cache poison bypassed with verified private bytes')
# Static adapter construction only: fake torch/api/graph; no run or numerical computation.
api=types.ModuleType('benchmarks.api');api.Implementation=lambda run:types.SimpleNamespace(run=run)
torch=types.ModuleType('torch')
seen=[]
private['graph'].make=lambda c,o: (seen.append((c.params,o)) or types.SimpleNamespace(implementation=types.SimpleNamespace(run=None),source_sha256={},cleanup=None))
with patch.dict(sys.modules,{'torch':torch,'benchmarks.api':api}):
 candidate=private['adapter'].make(case())
assert seen==[({'output':([8192,28672],'float32')},'MulFwdOp')]
checks.append('stub-only original adapter construction selects full Mul and FP32 output; no numeric execution')
# Simulate /proc maps without loading any shared library.
lines=[]
for path in pins:
 if path.endswith('.so'):
  st=Path(path).stat()
  lines.append(f'0000-0001 r--p 00000000 {os.major(st.st_dev):x}:{os.minor(st.st_dev):x} {st.st_ino} {path}')
with patch.object(Path,'read_text',return_value='\n'.join(lines)):m._loaded_vendor_pins(pins,require=True)
with patch.object(Path,'read_text',return_value=''):reject(lambda:m._loaded_vendor_pins(pins,require=True),RuntimeError)
with patch.object(Path,'read_text',return_value=lines[0]+' (deleted)'):reject(lambda:m._loaded_vendor_pins(pins),RuntimeError)
with patch.object(Path,'read_text',return_value=lines[0].replace('r--p','r--p').replace(str(Path(lines[0].split()[-1]).stat().st_ino),'0')):reject(lambda:m._loaded_vendor_pins(pins),RuntimeError)
checks.append('loaded vendor map path/hash/inode checks accept reviewed maps and reject missing/deleted/replaced mappings (synthetic maps)')
c=m.make(case());assert callable(c.implementation.run);c.cleanup()
with patch.object(Path,'read_text',return_value='{}'):reject(lambda:c.implementation.run(None,None),RuntimeError)
checks.append('lazy make/cleanup do not import runtime; unguarded execution rejected before import')
prior=dict(provider.PROVIDERS)
if not published:
 exec(compile((P/'registration.append.py').read_bytes(),str(P/'registration.append.py'),'exec'),provider.__dict__)
 for op,f in prior.items():
  if op!=m.KEY[1]:assert provider.PROVIDERS[op] is f
 sentinel=object()
 with patch.object(m,'make',return_value=sentinel):assert candidate_factory(case(),'ops-math')() is sentinel
 route=provider.PROVIDERS[m.KEY[1]]
 assert route.__defaults__[0] is prior.get(m.KEY[1])
 calls=[];old=lambda c:(calls.append(c.id) or sentinel)
 assert route(case(cid='unknown'),_previous=old) is sentinel and calls==['unknown']
 reject(lambda:route(case(cid='unknown'),_previous=None),UnsupportedCase)
 p=copy.deepcopy(m.PARAMS);p['out_dtype']='float32';reject(lambda:route(case(p),_previous=old),UnsupportedCase)
 assert calls==['unknown']
else:
 assert provider.PROVIDERS[m.KEY[1]].__name__=='_nn_dequant_registration1_route'
 sentinel=object()
 with patch.object(m,'make',return_value=sentinel):assert candidate_factory(case(),'ops-math')() is sentinel
checks.append('actual registry candidate_factory exact sentinel route; other op identities and unknown-case previous fallback retained' if not published else 'fresh published registry exact sentinel route')
assert not any(n in sys.modules for n in ('torch','torch_npu','acl','tvm','tilelang'))
result={'status':'passed','mode':'published' if published else 'private temporary in-process dispatch','checks':checks,'real_runtime_imports':0,'npu_runs':0,'numerical_validation':False,'accepted_added':0}
print(json.dumps(result,indent=2))
if not published:(P/'CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
