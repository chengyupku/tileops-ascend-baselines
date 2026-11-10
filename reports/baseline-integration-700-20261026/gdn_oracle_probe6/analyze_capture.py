# CPU-only post-analysis of captured payloads; no NPU imports or replay.
import os,sys,importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch_npu','acl','aclnn'):raise ImportError('CPU-only analysis forbids '+fullname)
sys.meta_path.insert(0,BlockDevice())
from pathlib import Path
import torch,json,hashlib
P=Path(__file__).resolve().parent; C=P/'runs/probe01--GDNFwdOp--prefill-narrow-state-bfloat16/output/capture'
torch.set_num_threads(4)
entries={p.stem:json.loads(p.read_text()) for p in C.glob('*.json')}
for name,e in entries.items():assert hashlib.sha256((C/(name+'.pt')).read_bytes()).hexdigest()==e['payload_sha256']
def bits(e):return [None if x is None else {k:x[k] for k in ('shape','dtype','bytes','sha256')} for x in e['tensors']]
base=bits(entries['reference_inputs_before'])
checks={name:bits(e)==base for name,e in entries.items() if 'inputs_' in name}
out=dict(input_preservation=checks,all_inputs_bitwise_identical=all(checks.values()),cpu_only=True,device_imports_blocked=True,outputs=[])
if 'reference_outputs' in entries and 'candidate_outputs' in entries:
 expected=torch.load(C/'reference_outputs.pt',map_location='cpu',weights_only=True)
 actual=torch.load(C/'candidate_outputs.pt',map_location='cpu',weights_only=True)
 for i,(a,b) in enumerate(zip(actual,expected,strict=True)):
  x,y=a.double(),b.double();diff=x-y;energy=(x*x+y*y).sum().item();err=(diff*diff).sum().item()
  out['outputs'].append(dict(index=i,elements=a.numel(),shape=list(a.shape),dtype=str(a.dtype),max_abs_error=diff.abs().max().item(),normalized_squared_error=err/energy if energy else 0,all_finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),tolerance_atol=0.016,tolerance_rtol=0.016,normalized_bound=0.001))
(P/'CAPTURE_ANALYSIS.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
