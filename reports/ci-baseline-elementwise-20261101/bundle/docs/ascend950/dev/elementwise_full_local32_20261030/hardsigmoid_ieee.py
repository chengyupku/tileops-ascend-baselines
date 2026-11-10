"""Exercise both SIMT tails and full SIMD tiles with IEEE/whole 16-bit domains."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import hardsigmoid as module
from tileops.ops.elementwise import HardsigmoidFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];op=HardsigmoidFwdOp();p=Path(module.__file__);frozen=hashlib.sha256(p.read_bytes()).hexdigest()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [64,65536,1048576]:
  if dtype!=torch.float32 and n>=65536:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).repeat(n//65536)
  else:
   f=torch.finfo(dtype)
   seed=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),f.max,-f.max,f.tiny,-f.tiny,f.smallest_normal/2,-f.smallest_normal/2,-3.,3.,-3.001,3.001,-2.999,2.999],dtype=dtype)
   x=seed.repeat((n+len(seed)-1)//len(seed))[:n]
  a=x.npu();before=a.cpu().view(torch.uint8).clone();got=op(a).cpu();ref=torch.nn.functional.hardsigmoid(x)
  compare_outputs(got,ref,ElementwiseWorkload('HardsigmoidFwdOp',(a,)).verification(a))
  assert torch.equal(before,a.cpu().view(torch.uint8))
  rows.append({'dtype':str(dtype),'n':n,'passed':True});print(rows[-1],flush=True)
assert hashlib.sha256(p.read_bytes()).hexdigest()==frozen
torch.npu.synchronize()
Path(__file__).with_name('hardsigmoid-ieee.json').write_text(json.dumps({'checks':rows,'module_sha256':frozen,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
