"""Floor scratch budget for large bias vectors and wide channel broadcasts."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_div as module
from tileops.ops.elementwise import DivFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from div_reference import div_reference
rows=[];torch.manual_seed(123)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for a_shape,b_shape in [((2,11008),(1,11008)),((2,16320),(1,16320)),((2,16384),(1,16384)),((1,4,8192),(4,1))]:
  x=torch.randn(a_shape).to(dtype);b=(torch.randn(b_shape)+.125).to(dtype)
  a=x.npu();c=b.npu();op=DivFwdOp(rounding_mode='floor')
  expected=div_reference(x,b,'floor');actual=op(a,c).cpu()
  compare_outputs(actual,expected,ElementwiseWorkload('DivFwdOp',(x,b),rounding_mode='floor').verification(x,b))
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  row={'dtype':str(dtype),'a_shape':a_shape,'b_shape':b_shape,'passed':True,'input_unchanged':True};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('div-ub-regression.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(__file__).with_name('div_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
