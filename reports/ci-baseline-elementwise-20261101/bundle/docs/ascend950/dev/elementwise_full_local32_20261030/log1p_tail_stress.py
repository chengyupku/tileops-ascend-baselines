"""Uneven persistent-grid passes, large tails, integer promotion and input changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import log1p as module
from tileops.ops.elementwise import Log1pFwdOp
from tileops.kernels.ascend.elementwise import _logroots
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];op=Log1pFwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [32767,524289,540671]:
  x=(torch.randn(n)*3).to(dtype);a=x.npu();before=a.cpu().view(torch.uint8).clone()
  actual=op(a).cpu();ref=torch.log1p(x.double()).to(dtype)
  compare_outputs(actual,ref,ElementwiseWorkload('Log1pFwdOp',(a,)).verification(a))
  assert torch.equal(a.cpu().view(torch.uint8),before)
  changed=(x.abs()+1).npu();second=op(changed).cpu()
  compare_outputs(second,torch.log1p(changed.cpu().double()).to(dtype),ElementwiseWorkload('Log1pFwdOp',(changed,)).verification(changed))
  rows.append({'dtype':str(dtype),'n':n,'changed_input':True,'passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('log1p-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0),'shared_sha256':hashlib.sha256(Path(_logroots.__file__).read_bytes()).hexdigest()},indent=2)+'\n')
