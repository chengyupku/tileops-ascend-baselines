import sys,types,pathlib,json,hashlib,ast,importlib.abc,importlib.util,difflib
P=pathlib.Path(__file__).resolve().parent; W=pathlib.Path('/opt/tileops-all-pairs/workspace'); T=W/'TileOPs'
sys.path[:0]=[str(P),str(T),str(T/'src'),str(W)]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('torch_npu','triton','sgl_kernel_npu')): raise RuntimeError('CPU device import blocked: '+fullname)
sys.meta_path.insert(0,Block())
import integration700_sgl_gdn_remote38 as adapter
from benchmarks.ascend_all_pairs import registry
from route_install import install
import benchmarks.ascend_all_pairs.sgl as real_provider
# Stub only runtime-heavy factory dependencies. Actual registry, provider and adapter execute.
torch=types.ModuleType('torch'); sys.modules['torch']=torch
api=types.ModuleType('benchmarks.api');api.Implementation=lambda fn:types.SimpleNamespace(run=fn);sys.modules['benchmarks.api']=api
src=adapter.ROOT/adapter.HOST
module=types.ModuleType('sgl_kernel_npu.fla.chunk');module.__file__=str(src)
exec(compile('def chunk_gated_delta_rule_native(*a,**kw): return None\ndef chunk_gated_delta_rule_npu(*a,**kw): return None',str(src),'exec'),module.__dict__)
sys.modules[module.__name__]=module
install(adapter); rows=[]
for op in ['GDNFwdOp','DeltaNetFwdOp']:
 c=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)),params={},inputs=[types.SimpleNamespace(shape=(1,1,1,128))]*(10 if op=='GDNFwdOp' else 7))
 candidate=registry.candidate_factory(c,'sgl-kernel-npu')();rows.append({'operator':op,'candidate':str(candidate),'run_module':candidate.implementation.run.__module__})
original=(T/'workloads/numerics.py').read_bytes(); prior=(P/'numerics.original.py').read_bytes(); assert original==prior
private=(P/'numerics_private.py').read_bytes(); dif=list(difflib.unified_diff(original.decode().splitlines(True),private.decode().splitlines(True),fromfile='original',tofile='private')); assert len([l for l in dif if l.startswith('-') and not l.startswith('---')])==1
(P/'DIAGNOSTIC_ONLY.patch').write_text(''.join(dif))
checker=T/'scripts/ascend_pair_check.py';(P/'CHECKER_PIN.json').write_text(json.dumps({'path':str(checker),'sha256':hashlib.sha256(checker.read_bytes()).hexdigest()},indent=2))
files=[src,adapter.ROOT/'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py',adapter.ROOT/'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so',checker,T/'workloads/numerics.py',T/'workloads/linear_attention/gdn.py']
(P/'CPU_PREFLIGHT.json').write_text(json.dumps({'passed':True,'runtime_imports_blocked':True,'numerical_validation':False,'interpreter':sys.executable,'registry':registry.__file__,'provider':real_provider.__file__,'dispatch':rows,'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'diagnostic_diff':dif},indent=2))
print('CPU_PREFLIGHT_PASSED')
