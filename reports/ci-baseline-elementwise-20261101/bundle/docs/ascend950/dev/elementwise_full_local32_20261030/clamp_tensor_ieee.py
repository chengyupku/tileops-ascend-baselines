import torch,torch_npu,itertools,json,hashlib
from pathlib import Path
from tileops.kernels.ascend.elementwise import clamp_tensor as module
from tileops.ops.elementwise import ClampTensorFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];source=Path(module.__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 values=[0.,-0.,-1.,1.,float('inf'),-float('inf'),float('nan')]
 seeds=[torch.tensor(v,dtype=dtype) for v in zip(*itertools.product(values,repeat=3))]
 op=ClampTensorFwdOp()
 for form in ['tail','large','broadcast']:
  if form=='broadcast':x,lo,hi=[torch.tensor(values,dtype=dtype).reshape(s) for s in [(7,1,1),(1,7,1),(1,1,7)]]
  else:
   n=343 if form=='tail' else 1048576
   x,lo,hi=[v.repeat((n+342)//343)[:n] for v in seeds]
  for mode in ['both','min','max']:
   cpu=(x,lo if mode!='max' else None,hi if mode!='min' else None)
   args=tuple(v.npu() if v is not None else None for v in cpu)
   before=[v.cpu().contiguous().view(torch.uint8).clone() if v is not None else None for v in args]
   ref=torch.clamp(*cpu);out=op(*args).cpu();compare_outputs(out,ref,ElementwiseWorkload('ClampTensorFwdOp',args).verification(*args))
   z=(ref==0)&(out==0);cpu_zero_sign=torch.equal(torch.signbit(out[z]),torch.signbit(ref[z]))
   # CPU scalar/vector loops disagree on signed-zero ties. Keep the original
   # Exact(atol=0, rtol=0) contract above, and require a stable tie-to-bound rule.
   rule=x
   if mode!='max':rule=torch.where(torch.isnan(lo)|(rule<=lo),lo,rule)
   if mode!='min':rule=torch.where(torch.isnan(hi)|(rule>=hi),hi,rule)
   z=(rule==0)&(out==0);assert torch.equal(torch.signbit(out[z]),torch.signbit(rule[z])),(dtype,form,mode,'tie-to-bound rule')
   for a,b in zip(args,before):
    if a is not None:assert torch.equal(a.cpu().contiguous().view(torch.uint8),b)
   rows.append({'dtype':str(dtype),'form':form,'mode':mode,'correct':True,'zero_sign_rule':True,'cpu_zero_sign':cpu_zero_sign,'input_unchanged':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
torch.npu.synchronize();Path(__file__).with_name('clamp-tensor-ieee.json').write_text(json.dumps({'rows':rows,'module_sha256':digest,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
