"""Force the full large-quotient repair across generic, bias, and widened channel paths."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_modulo as module
from tileops.ops.elementwise import FloorDivideFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from floor_divide_reference import floor_divide_reference
rows=[];torch.manual_seed(552)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 exponent={torch.float16:14,torch.bfloat16:17,torch.float32:23}[dtype]
 limit={torch.float16:8192,torch.bfloat16:65536,torch.float32:4194304}[dtype]
 for xs,bs in [((65537,),(65537,)),((5,960),(1,960)),((6,1088),(1,1088)),((2,64,128),(64,1))]:
  x=(1+torch.rand(xs)*.2).to(dtype);b=((.75+torch.rand(bs)*.5)*2**(-exponent)).to(dtype)
  x.flatten()[::2].neg_();b.flatten()[::3].neg_()
  assert bool(((x.double()/b.double()).abs()>limit).all())
  a=x.npu();c=b.npu();op=FloorDivideFwdOp()
  actual=op(a,c).cpu();expected=floor_divide_reference(x,b)
  compare_outputs(actual,expected,ElementwiseWorkload('FloorDivideFwdOp',(x,b)).verification(x,b))
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  row={'dtype':str(dtype),'a_shape':xs,'b_shape':bs,'passed':True,'all_quotients_above_fast_limit':True,'input_unchanged':True};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('floor_divide-repair.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name(name)):hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest() for name in ['_floor_divide.py','arithmetic_div.py','arithmetic.py']},'reference_sha256':hashlib.sha256(Path(__file__).with_name('floor_divide_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
