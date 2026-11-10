"""Odd channel counts with short and UB-exceeding per-channel spans."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import comparison as module
from tileops.ops.elementwise import LtFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from lt_reference import lt_reference
rows=[];torch.manual_seed(112)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for period in [3,19]:
  for repeat in [64,65536,65600]:
   x=(torch.rand(2,period,repeat)*.75+.75).to(dtype);b=(torch.rand(period,1)*4-2).to(dtype)
   x.reshape(-1)[::3]=b.expand_as(x).reshape(-1)[::3]
   a=x.npu();c=b.npu();actual=LtFwdOp()(a,c).cpu();expected=lt_reference(x,b)
   compare_outputs(actual,expected,ElementwiseWorkload('LtFwdOp',(x,b)).verification(x,b))
   assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
   row={'dtype':str(dtype),'period':period,'repeat':repeat,'passed':True,'input_unchanged':True};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('lt-channel-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name(name)):hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest() for name in []},'reference_sha256':hashlib.sha256(Path(__file__).with_name('lt_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
