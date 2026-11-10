"""Add broadcast/empty/unaligned boundaries, noncontiguous inputs and alpha changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic as module
from tileops.ops.elementwise import AddFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
DTYPES=[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]
SHAPES=[((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))]
rows=[]
def draw(shape,dtype):
 return torch.randn(shape).to(dtype) if dtype.is_floating_point else torch.randint(0,2 if dtype==torch.bool else 20,shape).to(dtype)
def verify(y,x,b,alpha):
 compare_outputs(y,torch.add(x,b,alpha=alpha),ElementwiseWorkload('AddFwdOp',(x,b),alpha=alpha).verification(x,b))
for dtype in DTYPES:
 op=AddFwdOp()
 for xs,bs in SHAPES:
  x=draw(xs,dtype);b=draw(bs,dtype)
  if x.ndim>=2 and x.numel():x=x.transpose(0,1).contiguous().transpose(0,1)
  a=x.npu();c=b.npu();op.alpha=1;y=op(a,c)
  assert y.shape==torch.broadcast_shapes(xs,bs) and y.dtype==dtype
  if y.numel():verify(y.cpu(),x,b,1)
  alpha=0 if dtype==torch.bool else .5 if dtype.is_floating_point else 2
  op.alpha=alpha;y=op(a,c)
  if y.numel():verify(y.cpu(),x,b,alpha)
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  rows.append({'dtype':str(dtype),'a_shape':xs,'b_shape':bs,'passed':True,'alpha_changed':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('add-boundaries.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
