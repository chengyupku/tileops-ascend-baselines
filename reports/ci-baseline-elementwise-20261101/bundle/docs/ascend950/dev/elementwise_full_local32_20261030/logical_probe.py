"""Independent truthiness oracle including NaNs, signed zero and raw bool bytes."""
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import logical_and
from tileops.ops.elementwise import LogicalAndFwdOp
rows=[]
for dtype in [torch.bool,torch.float16,torch.bfloat16,torch.float32,torch.int64]:
 for shape,other in [((128,),(128,)),((2,8,64),(8,1)),((7,),(1,))]:
  if dtype==torch.bool:
   x=torch.randint(0,256,shape,dtype=torch.uint8);y=torch.randint(0,256,other,dtype=torch.uint8)
   x.reshape(-1)[::3]=0;y.reshape(-1)[::2]=0
   a=x.npu().view(torch.bool);b=y.npu().view(torch.bool)
   ref=torch.logical_and(x!=0,y!=0)
  else:
   x=torch.randint(-3,4,shape).to(dtype);y=torch.randint(-3,4,other).to(dtype)
   if dtype.is_floating_point:
    vals=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),torch.finfo(dtype).tiny/2],dtype=dtype)
    x.reshape(-1)[:6]=vals
   else:x.reshape(-1)[0]=torch.iinfo(dtype).min
   a=x.npu();b=y.npu();ref=torch.logical_and(x,y)
  op=LogicalAndFwdOp();out=op(a,b);torch.npu.synchronize()
  torch.testing.assert_close(out.cpu(),ref,rtol=0,atol=0)
  torch.testing.assert_close(out.view(torch.uint8).cpu(),ref.to(torch.uint8),rtol=0,atol=0)
  rows.append({'dtype':str(dtype),'shape':shape,'other':other,'passed':True})
  Path(__file__).with_name('logical-and-probe-progress.json').write_text(json.dumps(rows))
  print(rows[-1],flush=True)
Path(__file__).with_name('logical-and-probe.json').write_text(json.dumps({'checks':rows,'runtime_limit':torch.npu.get_device_limit(0)},indent=2))
