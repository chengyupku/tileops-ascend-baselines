"""PReLU uneven channel spans, multi-iteration tails, and changing both inputs."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from tileops.kernels.ascend.elementwise import prelu as module
from tileops.ops.elementwise import PreluFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
def check(y,x,w):
 compare_outputs(y,F.prelu(x.double(),w.double()).to(x.dtype),ElementwiseWorkload('PreluFwdOp',(x,w)).verification(x,w))
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for shape in [(2,3,32767),(3,5,785),(1,17,1025)]:
  x=torch.randn(shape).to(dtype);w=torch.linspace(-.5,1.5,shape[1]).to(dtype)
  op=PreluFwdOp();a=x.npu();b=w.npu();actual=op(a,b).cpu();check(actual,x,w)
  assert torch.equal(a.cpu(),x) and torch.equal(b.cpu(),w)
  second_x=x.neg()+.25;second_w=w.neg()+.125
  second=op(second_x.npu(),second_w.npu()).cpu();check(second,second_x,second_w)
  rows.append({'dtype':str(dtype),'shape':shape,'passed':True,'changed_inputs':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('prelu-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
