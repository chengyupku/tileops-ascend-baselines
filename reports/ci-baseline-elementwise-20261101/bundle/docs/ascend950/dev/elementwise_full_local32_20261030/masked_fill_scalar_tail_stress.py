"""Large uneven masked-fill tiles with all dtype widths and changing inputs."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import masked_fill_scalar as module
from tileops.ops.elementwise import MaskedFillScalarFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
def check(y,x,m,v):
 compare_outputs(y,x.masked_fill(m,v),ElementwiseWorkload('MaskedFillScalarFwdOp',(x,m),value=v.item()).verification(x,m))
for dtype in [torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]:
 for n in [32767,524289,540671]:
  x=(torch.randn(n).to(dtype) if dtype.is_floating_point else torch.randint(0,2 if dtype==torch.bool else 100,(n,)).to(dtype))
  m=torch.randint(0,2,(n,)).bool();v=torch.tensor(False if dtype==torch.bool else 7,dtype=dtype)
  a=x.npu();b=m.npu();op=MaskedFillScalarFwdOp(value=v.item());check(op(a,b).cpu(),x,m,v)
  assert torch.equal(a.cpu(),x) and torch.equal(b.cpu(),m)
  x2=~x if dtype==torch.bool else x+1;m2=~m;v2=torch.tensor(True if dtype==torch.bool else 3,dtype=dtype)
  a.copy_(x2.npu());b.copy_(m2.npu());op.value=v2.item();check(op(a,b).cpu(),x2,m2,v2)
  assert torch.equal(a.cpu(),x2) and torch.equal(b.cpu(),m2)
  rows.append({'dtype':str(dtype),'n':n,'passed':True,'changed_inputs':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('masked-fill-scalar-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
