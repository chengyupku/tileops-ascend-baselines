import sys, os, pathlib, ast, json, time, hashlib, types
assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
sys.dont_write_bytecode=True
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','tilelang','workloads','benchmarks'):raise RuntimeError('forbidden import '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,'/root/dyq/workspace/.venv/lib/python3.12/site-packages')
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
p=pathlib.Path('/root/dyq/workspace/TileOPs/workloads/gemm.py');src=p.read_text();tree=ast.parse(src)
fns={}
for cls in tree.body:
 if isinstance(cls,ast.ClassDef) and cls.name in ('BmmWorkload','GroupedGemmWorkload'):
  method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='ref_program')
  ns={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),'exec'),ns);fns[cls.name]=ns['ref_program']
print(json.dumps({'event':'env','torch':torch.__version__,'config':torch.__config__.show(),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'cpu_flags':next(x for x in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('flags')),'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}),flush=True)
def tensor(shape):
 n=1
 for x in shape:n*=x
 return (((torch.arange(n,dtype=torch.int32)%19)-9).to(torch.float16)/16).reshape(shape)
for name in ['bmm','fwd','dgrad']:
 a=tensor((2,512,512)) if name=='bmm' else tensor((512,512));b=tensor((2,512,512)) if name=='bmm' else tensor((1,512,512))
 obj=types.SimpleNamespace(transpose_a=False,transpose_b=name=='fwd')
 args=(a,b) if name=='bmm' else (a,b,torch.tensor([512],dtype=torch.int32),torch.tensor([0],dtype=torch.int32))
 fn=fns['BmmWorkload' if name=='bmm' else 'GroupedGemmWorkload']
 start=time.perf_counter();cpu=time.process_time()
 for _ in range(3):out=fn(obj,*args)
 print(json.dumps({'event':'op','name':name,'repeats':3,'wall_s':time.perf_counter()-start,'cpu_s':time.process_time()-cpu,'shape':[list(a.shape),list(b.shape)],'stride':[list(a.stride()),list(b.stride())],'output_shape':list(out.shape),'output_hash':hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}),flush=True)
libs=sorted(set(l.split()[-1] for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if '.so' in l));forbidden=[s for s in libs if any(x in s.lower() for x in ['libtorch_npu','libascend','libhccl','libacl_op'])]
print(json.dumps({'event':'maps','libraries':libs,'forbidden':forbidden,'forbidden_imports':[s for s in sys.modules if s.split('.')[0] in ('torch_npu','acl')]}),flush=True)
assert not forbidden
