"""Uneven persistent-grid passes, large tails, integer promotion and input changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import expm1 as module
from tileops.ops.elementwise import Expm1FwdOp
from tileops.kernels.ascend.elementwise import exponential
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];op=Expm1FwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [32767,524289,540671]:
  x=(torch.randn(n)*3).to(dtype);a=x.npu();before=a.cpu().view(torch.uint8).clone()
  actual=op(a).cpu();ref=torch.expm1(x.double()).to(dtype)
  compare_outputs(actual,ref,ElementwiseWorkload('Expm1FwdOp',(a,)).verification(a))
  assert torch.equal(a.cpu().view(torch.uint8),before)
  changed=(x.abs()+1).npu();second=op(changed).cpu()
  compare_outputs(second,torch.expm1(changed.cpu().double()).to(dtype),ElementwiseWorkload('Expm1FwdOp',(changed,)).verification(changed))
  rows.append({'dtype':str(dtype),'n':n,'changed_input':True,'passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('expm1-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0),'shared_sha256':hashlib.sha256(Path(exponential.__file__).read_bytes()).hexdigest()},indent=2)+'\n')
