"""Remainder boundaries with bidirectional broadcasts and changed noncontiguous inputs."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_modulo as module
from tileops.ops.elementwise import RemainderFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from remainder_reference import remainder_reference
DTYPES=[torch.float16,torch.bfloat16,torch.float32]
SHAPES=[((),()),((0,3),(1,3)),((65,),(65,)),((129,),()),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((16387,),(16387,)),((2,4,64),(4,1)),((2,3,65),(3,1))]
rows=[]
def draw(shape,dtype):
 return torch.randn(shape).to(dtype) if dtype.is_floating_point else torch.randint(0,2 if dtype==torch.bool else 20,shape).to(dtype)
def verify(y,x,b):
 compare_outputs(y,remainder_reference(x,b),ElementwiseWorkload('RemainderFwdOp',(x,b)).verification(x,b))
for dtype in DTYPES:
 op=RemainderFwdOp()
 for xs,bs in SHAPES:
  x=draw(xs,dtype);b=draw(bs,dtype)
  if x.ndim>=2 and x.numel():x=x.transpose(0,1).contiguous().transpose(0,1)
  a=x.npu();c=b.npu();y=op(a,c)
  assert y.shape==torch.broadcast_shapes(xs,bs) and y.dtype==dtype
  if y.numel():verify(y.cpu(),x,b)
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  x2=~x if dtype==torch.bool else x+1;b2=~b if dtype==torch.bool else b+2
  a.copy_(x2.npu());c.copy_(b2.npu());y=op(a,c)
  if y.numel():verify(y.cpu(),x2,b2)
  assert torch.equal(a.cpu(),x2) and torch.equal(c.cpu(),b2)
  rows.append({'dtype':str(dtype),'a_shape':xs,'b_shape':bs,'passed':True,'changed_inputs':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('remainder-boundaries.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(Path(__file__).with_name('remainder_reference.py').read_bytes()).hexdigest(),'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
