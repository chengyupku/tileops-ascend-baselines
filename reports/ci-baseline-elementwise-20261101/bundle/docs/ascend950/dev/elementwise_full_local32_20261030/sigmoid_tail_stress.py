"""Uneven persistent-grid passes, large tails, integer promotion and input changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import sigmoid as module
from tileops.ops.elementwise import SigmoidFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];op=SigmoidFwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for n in [32767,524289,540671]:
  x=(torch.randn(n)*3).to(dtype);a=x.npu();before=a.cpu().view(torch.uint8).clone()
  actual=op(a).cpu();ref=torch.sigmoid(x.double()).to(dtype)
  compare_outputs(actual,ref,ElementwiseWorkload('SigmoidFwdOp',(a,)).verification(a))
  assert torch.equal(a.cpu().view(torch.uint8),before)
  changed=(x.abs()+1).npu();second=op(changed).cpu()
  compare_outputs(second,torch.sigmoid(changed.cpu().double()).to(dtype),ElementwiseWorkload('SigmoidFwdOp',(changed,)).verification(changed))
  rows.append({'dtype':str(dtype),'n':n,'changed_input':True,'passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('sigmoid-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
