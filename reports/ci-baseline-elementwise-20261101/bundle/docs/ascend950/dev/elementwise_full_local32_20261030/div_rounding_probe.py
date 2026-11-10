"""Triangulate current Div against CPU and NPU references without changing tolerances."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_div as module
from tileops.ops.elementwise import DivFwdOp
rows=[];torch.manual_seed(123)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for mode in ['trunc','floor']:
  for pattern in ['ordinary','special']:
   if pattern=='ordinary':
    x=(torch.randn(16384)*10).to(dtype);b=(torch.randn(16384)*3).to(dtype)
   else:
    vals=torch.tensor([0.,-0.,1.,-1.,.1,-2.5,7.,3e4,float('inf'),-float('inf'),float('nan'),1e-38,-1e-38],dtype=dtype)
    pairs=torch.cartesian_prod(vals,vals);x=pairs[:,0].contiguous();b=pairs[:,1].contiguous()
   a=x.npu();c=b.npu();actual=DivFwdOp(rounding_mode=mode)(a,c).cpu();cpu=torch.div(x,b,rounding_mode=mode);npu=torch.div(a,c,rounding_mode=mode).cpu()
   def diff(y,z):return ~((y==z)|(y.isnan()&z.isnan()))
   bad=diff(actual,cpu)|diff(actual,npu)|diff(cpu,npu);ids=bad.nonzero().flatten()[:20].tolist()
   row={'dtype':str(dtype),'mode':mode,'pattern':pattern,'kernel_cpu_mismatches':int(diff(actual,cpu).sum()),'kernel_npu_mismatches':int(diff(actual,npu).sum()),'cpu_npu_mismatches':int(diff(cpu,npu).sum()),'examples':[{'a':float(x[i]),'b':float(b[i]),'kernel':float(actual[i]),'cpu':float(cpu[i]),'npu':float(npu[i])} for i in ids]}
   rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('div-rounding-probe.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
