"""Exhaustive 16-bit truthiness and raw bool bytes through the public operator."""
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import logical_and
from tileops.ops.elementwise import LogicalAndFwdOp
checks=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.bool]:
 if dtype==torch.bool:
  raw=torch.arange(256,dtype=torch.uint8).repeat_interleave(256);other=torch.arange(256,dtype=torch.uint8).repeat(256)
  x=raw.view(torch.bool);y=other.view(torch.bool);ref=torch.logical_and(raw!=0,other!=0)
 else:
  if dtype==torch.float32:
   bits=torch.tensor([0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xffffffff,0x3f800000,0xbf800000,0x7f7fffff],dtype=torch.int64).to(torch.int32).repeat(4096)
   x=bits.view(dtype)
  else:x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
  y=torch.ones_like(x);ref=torch.logical_and(x,y)
 for zeros in [False,True]:
  yy=y.clone()
  if zeros:yy[::3]=0
  if dtype==torch.bool:ref=torch.logical_and(raw!=0,yy.view(torch.uint8)!=0)
  else:ref=torch.logical_and(x,yy)
  a=x.npu();b=yy.npu();out=LogicalAndFwdOp()(a,b);torch.npu.synchronize()
  torch.testing.assert_close(out.cpu(),ref,rtol=0,atol=0)
  torch.testing.assert_close(out.view(torch.uint8).cpu(),ref.to(torch.uint8),rtol=0,atol=0)
  checks.append({'dtype':str(dtype),'n':x.numel(),'zeros':zeros,'passed':True});print(checks[-1],flush=True)
Path(__file__).with_name('logical-and-truth-edges.json').write_text(json.dumps({'checks':checks,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
