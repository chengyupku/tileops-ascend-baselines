"""Cartesian IEEE maximum edge values across same-shape, bias, and channel layouts."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_extrema as module
from tileops.ops.elementwise import MaximumFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from maximum_reference import maximum_reference
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 info=torch.finfo(dtype)
 values=torch.tensor([0.,-0.,1.,-1.,2.,-2.,.5,-.5,3.,-3.,float('inf'),-float('inf'),float('nan'),info.tiny,-info.tiny,info.max,-info.max,info.smallest_normal*info.eps,-info.smallest_normal*info.eps],dtype=dtype)
 n=values.numel()
 for layout in ['same','bias','channel']:
  if layout=='same':x=values.repeat_interleave(n);b=values.repeat(n)
  elif layout=='bias':x=values[:,None].repeat(1,n);b=values[None,:]
  else:x=values[:,None,None].repeat(1,n,64);b=values[:,None]
  a=x.npu();c=b.npu();actual=MaximumFwdOp()(a,c).cpu();expected=maximum_reference(x,b)
  compare_outputs(actual,expected,ElementwiseWorkload('MaximumFwdOp',(x,b)).verification(x,b))
  zero=(actual==0)&(expected==0)
  assert torch.equal(actual.signbit()[zero],expected.signbit()[zero])
  storage=torch.int16 if dtype.itemsize==2 else torch.int32
  assert torch.equal(a.cpu().view(storage),x.view(storage)) and torch.equal(c.cpu().view(storage),b.view(storage))
  row={'dtype':str(dtype),'layout':layout,'pairs':n*n,'passed':True,'signed_zero':True,'input_unchanged':True};rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('maximum-special-values.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name('arithmetic.py')):hashlib.sha256(Path(module.__file__).with_name('arithmetic.py').read_bytes()).hexdigest()},'reference_sha256':hashlib.sha256(Path(__file__).with_name('maximum_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
