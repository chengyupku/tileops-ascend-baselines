"""PReLU all 16-bit patterns and FP32 random bits on generic and SIMD paths."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from tileops.kernels.ascend.elementwise import prelu as module
from tileops.ops.elementwise import PreluFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];torch.manual_seed(42)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 if dtype!=torch.float32:
  base=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
 else:
  base=torch.randint(-2147483648,2147483647,(65536,),dtype=torch.int32).view(dtype)
  base[:12]=torch.tensor([0.,-0.,1.,-1.,float('inf'),-float('inf'),float('nan'),1e-38,-1e-38,1e38,-1e38,1e-45])
 for layout in ['generic_tail','channel_simd']:
  x=torch.cat([base,torch.tensor([-0.],dtype=dtype)]) if layout=='generic_tail' else base.reshape(1,16,4096)
  a=x.npu();before=a.cpu().reshape(-1).view(torch.uint8).clone();op=PreluFwdOp()
  for slope in [0.,-0.,.1,-1.,2.,float('inf'),-float('inf'),float('nan')]:
   w=torch.tensor([slope],dtype=dtype);b=w.npu();w_before=b.cpu().view(torch.uint8).clone()
   actual=op(a,b).cpu();ref=F.prelu(x.double(),w.double()).to(dtype)
   compare_outputs(actual,ref,ElementwiseWorkload('PreluFwdOp',(x,w)).verification(x,w))
   zeros=(x==0)&(ref==0)
   assert torch.equal(torch.signbit(actual[zeros]),torch.signbit(ref[zeros])),'zero sign changed'
   assert torch.equal(a.cpu().reshape(-1).view(torch.uint8),before)
   assert torch.equal(b.cpu().view(torch.uint8),w_before)
   rows.append({'dtype':str(dtype),'layout':layout,'slope':str(slope),'values':x.numel(),'passed':True,'input_unchanged':True,'signed_zero':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('prelu-numerical-final.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
