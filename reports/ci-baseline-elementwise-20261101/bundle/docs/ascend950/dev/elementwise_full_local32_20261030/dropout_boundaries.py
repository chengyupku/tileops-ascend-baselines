"""Dropout bounds, strides, replay, scaling, and explicit-output canaries."""
import json,hashlib
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import dropout as module
from tileops.ops.elementwise import DropoutFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
def check(got,x,p):
 evidence=ElementwiseWorkload('DropoutFwdOp',(x,),p=p).verification(x)
 return compare_outputs(got,x/(1-p),evidence)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 op=DropoutFwdOp(p=.3,seed=42)
 for shape in [(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,)]:
  x=torch.randn(shape).to(dtype)
  if len(shape)==2 and x.numel():x=x.t().contiguous().t()
  a=x.npu();before=a.cpu().contiguous().reshape(-1).view(torch.uint8).clone();out=op(a);torch.npu.synchronize()
  assert out.shape==x.shape and out.dtype==x.dtype
  if x.numel():
   check(out.cpu(),x,.3)
   assert torch.equal(out,op(a)), 'same seed replay'
   assert out.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
  assert torch.equal(before,a.cpu().contiguous().reshape(-1).view(torch.uint8))
  rows.append({'dtype':str(dtype),'shape':shape,'correct':True,'replay':True,'input_unchanged':True});print(rows[-1],flush=True)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 n=16387;k=module.AscendDropoutKernel(n,dtype,.3,42)
 explicit=tilelang.compile(k._compiled_fn.prim_func,target='ascend',out_idx=[])
 x=torch.ones(n,dtype=dtype);a=x.npu();guarded=torch.full((n+128,),321,device='npu',dtype=dtype);output=guarded[64:64+n];output.zero_()
 explicit(a,output);torch.npu.synchronize()
 assert bool((guarded[:64]==321).all()) and bool((guarded[64+n:]==321).all())
 actual=output.cpu();check(actual,x,.3)
 bad=actual.clone();bad[-1]=123
 try:check(bad,x,.3)
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'check':'sentinel_and_negative_control','correct':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('dropout-boundaries.json').write_text(json.dumps({'rows':rows,'runtime_limit':torch.npu.get_device_limit(0),'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()},indent=2)+'\n')
