import sys,os,time,json,hashlib,pathlib,platform
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='4'
os.environ['ONEDNN_VERBOSE']='1'
sys.dont_write_bytecode=True
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl'):raise RuntimeError('NPU import forbidden '+fullname)
sys.meta_path.insert(0,Block())
sys.path[:0]=['/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
import torch
torch.set_num_threads(4)
P=pathlib.Path(__file__).parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def tensor(shape):
 n=1
 for d in shape:n*=d
 return ((torch.arange(n,dtype=torch.int32)%19)-9).to(torch.float16).reshape(shape)/16
def th():
 out={}
 for p in pathlib.Path('/proc/self/task').glob('*/stat'):
  s=p.read_text().split();out[p.parent.name]=int(s[13])+int(s[14])
 return out
print(json.dumps({'event':'env','host':platform.node(),'torch':torch.__version__,'config':torch.__config__.show(),'cpuinfo':next((l for l in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('Features')),''),'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'script_sha':sha(__file__),'torch_init_sha':sha(torch.__file__)}),flush=True)
for name in ['bmm','fwd','dgrad','wgrad']:
 a=tensor((2,128,128)) if name=='bmm' else tensor((128,128))
 b=tensor(tuple(a.shape))
 if name=='fwd':b=b.T.contiguous()
 if name=='wgrad':a=a.T
 fn=torch.bmm if name=='bmm' else torch.mm
 before=th();c0=time.process_time();t0=time.perf_counter()
 for i in range(3):out=fn(a,b)
 elapsed=time.perf_counter()-t0;cpu=time.process_time()-c0;after=th()
 print(json.dumps({'event':'op','name':name,'shape':[list(a.shape),list(b.shape)],'stride':[list(a.stride()),list(b.stride())],'dtype':str(a.dtype),'input_hashes':[hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for x in [a,b]],'output_hash':hashlib.sha256(out.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),'repeats':3,'wall_s':elapsed,'cpu_s':cpu,'active_thread_ticks':{k:v-before.get(k,0) for k,v in after.items() if v>before.get(k,0)}}),flush=True)
libs=sorted(set(l.split()[-1] for l in pathlib.Path('/proc/self/maps').read_text().splitlines() if '.so' in l))
print(json.dumps({'event':'maps','libraries':libs,'forbidden':[s for s in libs if any(v in s.lower() for v in ['libtorch_npu','libascendcl','libhccl'])],'blocked_imports_loaded':[s for s in sys.modules if s.split('.')[0] in ['torch_npu','acl']]}),flush=True)
