"""Uneven persistent-grid passes, large tails, integer promotion and input changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import reciprocal as module
from tileops.ops.elementwise import ReciprocalFwdOp
rows=[];op=ReciprocalFwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [32767,524289,540671]:
  x=(torch.randn(n)*3).to(dtype);a=x.npu();before=a.cpu().view(torch.uint8).clone()
  actual=op(a).cpu();ref=torch.reciprocal(x.double()).to(dtype)
  tol=1e-5 if dtype==torch.float32 else 1e-3 if dtype==torch.float16 else 1.6e-2
  torch.testing.assert_close(actual,ref,rtol=tol,atol=tol,equal_nan=True)
  assert torch.equal(a.cpu().view(torch.uint8),before)
  changed=(x.abs()+1).npu();second=op(changed).cpu()
  torch.testing.assert_close(second,torch.reciprocal(changed.cpu().double()).to(dtype),rtol=tol,atol=tol)
  rows.append({'dtype':str(dtype),'n':n,'changed_input':True,'passed':True});print(rows[-1],flush=True)
for dtype in [torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64]:
 info=torch.iinfo(dtype)
 for shape in [(0,3),(),(129,)]:
  x=torch.tensor([info.min,info.max,0,1,-1 if info.min<0 else 128],dtype=dtype)
  if shape==():x=x[1]
  elif shape==(0,3):x=x[:0].reshape(shape)
  else:x=x.repeat(26)[:129]
  a=x.npu();actual=op(a).cpu();ref=torch.reciprocal(x.float())
  assert actual.dtype==torch.float32 and actual.shape==x.shape
  torch.testing.assert_close(actual,ref,rtol=1e-5,atol=1e-5,equal_nan=True)
  rows.append({'dtype':str(dtype),'shape':shape,'passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('reciprocal-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
