import os,sys,json,hashlib,time,resource,ast,traceback
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['OMP_NUM_THREADS']='4'; os.environ['MKL_NUM_THREADS']='4'; os.environ['OPENBLAS_NUM_THREADS']='4'
class BlockNpu:
 def find_spec(self, fullname, path=None, target=None):
  if fullname=='torch_npu' or fullname.startswith('torch_npu.'):
   raise RuntimeError('CPU_ONLY_IMPORT_BLOCKED')
sys.meta_path.insert(0,BlockNpu())
import torch
import numpy as np
P=Path(__file__).resolve().parent
# load only private function from exact proposal source, avoiding unrelated workload imports
src=(P/'gemm_cpu_bf16.py').read_text(); mod=ast.parse(src)
node=next(n for n in mod.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_fp32_reduction')
code=compile(ast.Module(body=[node], type_ignores=[]),str(P/'gemm_cpu_bf16.py'),'exec')
gl={'torch':torch}; exec(code,gl); patched=gl['matmul_fp32_reduction']
def load(name):
 raw=(P/(name+'.bin')).read_bytes(); return torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).reshape(1024,1024),hashlib.sha256(raw).hexdigest()
a,ha=load('candidate_0_input0'); b,hb=load('candidate_0_input1'); cap,hc=load('candidate_0_output'); ref,hr=load('reference_0_output')
torch.set_num_threads(4)
# warm then timed runs; retain exact captured inputs
with torch.autocast(device_type='cpu',enabled=False):
 orig=torch.matmul(a,b)
 _=patched(a,b)
for _ in range(2):
 with torch.autocast(device_type='cpu',enabled=False): torch.matmul(a,b)
 patched(a,b)
def timed(fn):
 t=[]
 for _ in range(5):
  t0=time.perf_counter(); x=fn(); t.append((time.perf_counter()-t0)*1e3)
 return x,t
orig,ot=timed(lambda: torch.matmul(a,b))
fix,ft=timed(lambda: patched(a,b))
f64=a.double()@b.double(); f64r=f64.to(torch.bfloat16)
def cmp(x,y):
 d=(x.double()-y.double()).abs(); lim=.016+.016*y.double().abs(); bad=d>lim
 return {'violations':int(bad.sum()),'max_abs':float(d.max()),'mean_abs':float(d.mean()),'unequal':int((x!=y).sum()),'isclose':bool(torch.allclose(x,y,atol=.016,rtol=.016))}
def sha(x): return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
maps=Path('/proc/self/maps').read_text(); forbidden=[l for l in maps.splitlines() if any(k in l.lower() for k in ('libtorch_npu','libascendcl','libhccl'))]
out={'cpu_only':True,'torch':torch.__version__,'torch_config':torch.__config__.show(),'threads':torch.get_num_threads(),'env':{k:os.environ.get(k) for k in ['TORCH_DEVICE_BACKEND_AUTOLOAD','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'shape':[1024,1024],'dtype':str(a.dtype),'input_hashes':{'a':ha,'b':hb,'captured_candidate':hc,'captured_reference':hr},'output_hashes':{'local_original':sha(orig),'patched':sha(fix),'fp64_rounded':sha(f64r)},'comparisons':{'local_original_vs_captured_reference':cmp(orig,ref),'patched_vs_captured_reference':cmp(fix,ref),'local_original_vs_fp64':cmp(orig,f64r),'patched_vs_fp64':cmp(fix,f64r),'captured_candidate_vs_patched':cmp(cap,fix)},'timing_ms':{'local_original':ot,'patched':ft,'local_original_median':float(np.median(ot)),'patched_median':float(np.median(ft))},'rss_kb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'maps_forbidden':forbidden,'torch_npu_imported':'torch_npu' in sys.modules,'return_code':0}
(P/'ARM_RESULT.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
