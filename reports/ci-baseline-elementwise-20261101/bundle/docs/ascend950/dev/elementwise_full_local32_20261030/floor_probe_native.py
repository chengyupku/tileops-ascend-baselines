"""Exact CPU-oracle checks, including every FP16/BF16 bit pattern."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import floor as module
from tileops.ops.elementwise import FloorFwdOp
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64]:
 if dtype in (torch.float16,torch.bfloat16):
  base=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
 elif dtype==torch.float32:
  base=torch.tensor([0.,-0.,1e-45,-1e-45,1e-38,-1e-38,1.,-1.,float('inf'),-float('inf'),float('nan')],dtype=dtype)
 else:
  info=torch.iinfo(dtype);base=torch.tensor([info.min,info.max,0,1,-1 if info.min<0 else 255],dtype=dtype)
 for n in [7,128,65537]:
  x=base.repeat((n+len(base)-1)//len(base))[:n].clone();a=x.npu();op=FloorFwdOp();y=op(a);ref=torch.floor(x)
  torch.testing.assert_close(y.cpu(),ref,rtol=0,atol=0,equal_nan=True)
  if dtype.is_floating_point:assert torch.equal(torch.signbit(y.cpu()[~torch.isnan(ref)]),torch.signbit(ref[~torch.isnan(ref)]))
  assert torch.equal(a.cpu().view(torch.uint8),x.view(torch.uint8))
  assert y.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
  rows.append({'dtype':str(dtype),'n':n,'passed':True});print(rows[-1],flush=True)
  Path(__file__).with_name('floor-probe-native-progress.json').write_text(json.dumps(rows))
torch.npu.synchronize()
Path(__file__).with_name('floor-probe-native.json').write_text(json.dumps({'checks':rows,'runtime_limit':torch.npu.get_device_limit(0),'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()},indent=2)+'\n')
