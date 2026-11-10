"""Full 16-bit patterns, float32 edges and raw bool truthiness for logical NOT."""
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import isnan
from tileops.ops.elementwise import IsnanFwdOp
checks=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.bool]:
 if dtype==torch.bool:
  raw=torch.arange(256,dtype=torch.uint8).repeat(256);x=raw.view(torch.bool)
 elif dtype==torch.float32:
  bits=torch.tensor([0,0x80000000,1,0x80000001,0x7fffff,0x807fffff,0x800000,0x80800000,0x7f800000,0xff800000,0x7f800001,0x7fc00000,0xffffffff,0x3f800000,0xbf800000,0x7f7fffff],dtype=torch.int64).to(torch.int32).repeat(4096);x=bits.view(dtype)
 else:x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
 for n in [1,7,129,65536]:
  v=x[:n];ref=torch.isnan(v) if dtype!=torch.bool else torch.zeros(n,dtype=torch.bool)
  a=v.npu();out=IsnanFwdOp()(a);torch.npu.synchronize()
  torch.testing.assert_close(out.cpu(),ref,rtol=0,atol=0)
  torch.testing.assert_close(out.view(torch.uint8).cpu(),ref.to(torch.uint8),rtol=0,atol=0)
  checks.append({'dtype':str(dtype),'n':n,'passed':True});print(checks[-1],flush=True)
Path(__file__).with_name('isnan-truth-edges.json').write_text(json.dumps({'checks':checks,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
