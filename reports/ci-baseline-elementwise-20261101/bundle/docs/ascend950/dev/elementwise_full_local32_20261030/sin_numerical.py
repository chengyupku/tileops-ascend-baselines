"""Independent CPU FP64 oracle across storage bit patterns and FP32 exponents."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import sin as module
from tileops.ops.elementwise import SinFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];torch.manual_seed(20261030)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 if dtype!=torch.float32:
  x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
 else:
  bits=torch.randint(0,2**32,(65536,),dtype=torch.int64).to(torch.int32)
  x=bits.view(torch.float32)
  x=torch.cat([x,torch.tensor([0.,-0.,1e-45,-1e-45,1e-38,-1e-38,1e10,-1e10,1e20,-1e20,3.4e38,-3.4e38,float('inf'),-float('inf'),float('nan')])])
 # Add a tail to the complete low-precision bit-pattern set.
 x=torch.cat([x,x[:1]]);a=x.npu();before=a.cpu().view(torch.uint8).clone()
 y=SinFwdOp()(a).cpu();ref=torch.sin(x.double()).to(dtype)
 error=None
 try:
  compare_outputs(y,ref,ElementwiseWorkload('SinFwdOp',(a,)).verification(a))
  zero=ref==0;assert torch.equal(torch.signbit(y[zero]),torch.signbit(ref[zero]))
 except AssertionError as e:error=str(e)
 assert torch.equal(a.cpu().view(torch.uint8),before)
 diff=(y.float()-ref.float()).abs();indices=torch.argsort(diff.nan_to_num(nan=-1),descending=True)[:8]
 rows.append({'dtype':str(dtype),'values':len(x),'passed':error is None,'error':error,'worst':[{'input':str(x[i].item()),'actual':str(y[i].item()),'cpu_fp64':str(ref[i].item())} for i in indices]})
 print(rows[-1],flush=True)
 Path(__file__).with_name('sin-numerical.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
 if error:raise AssertionError(error)
torch.npu.synchronize()
