import os,sys,pathlib,json
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
p=pathlib.Path(__file__).resolve().parent;sys.path[:0]=[str(p/'source/src'),str(p/'source'),str(p)]
import torch
from workloads.attention.mla import MLADenseBwdWorkload,MLAVarlenBwdWorkload
from workloads.device import set_run_device
from workloads.numerics import compare_outputs
from adapters.mla_backward_saved import reference
set_run_device('cpu');torch.set_num_threads(2);results=[]
for dtype in (torch.float32,torch.bfloat16,torch.float16):
 for causal in (False,True):
  for w in (MLADenseBwdWorkload(2,7,5,2,4,3,dtype,causal,.4),MLAVarlenBwdWorkload([7,3],[5,8],2,4,3,dtype,causal,.4)):
   a=w.gen_inputs();expected=w.ref_program(*a);actual=reference(*a,is_causal=causal,sm_scale=.4);check=compare_outputs(actual,expected,w.verification(*a));assert check.checked_outputs
   z=reference(torch.zeros_like(a[0]),*a[1:],is_causal=causal,sm_scale=.4);assert all(torch.count_nonzero(t)==0 for t in z)
   results.append({'kind':type(w).__name__,'dtype':str(dtype),'causal':causal,'passed':True})
(p/'MLA_SAVED_CPU_TEST.json').write_text(json.dumps(results,indent=2));print('passed',len(results))
