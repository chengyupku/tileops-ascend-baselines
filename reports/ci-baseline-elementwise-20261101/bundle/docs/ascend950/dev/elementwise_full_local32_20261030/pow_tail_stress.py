"""Uneven Pow tiles, changing inputs, and explicit compiled-output sentinels."""
import hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import arithmetic_power as module
from tileops.ops.elementwise import PowFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from pow_reference import pow_reference
DTYPES=[torch.float16,torch.bfloat16,torch.float32]
rows=[]
def check(y,x,b):
 compare_outputs(y,pow_reference(x,b),ElementwiseWorkload('PowFwdOp',(x,b)).verification(x,b))
for dtype in DTYPES:
 for n in [32767,524289,540671]:
  x=torch.randn(n).to(dtype) if dtype.is_floating_point else torch.randint(0,2 if dtype==torch.bool else 20,(n,)).to(dtype)
  b=x.roll(17);a=x.npu();c=b.npu();op=PowFwdOp();check(op(a,c).cpu(),x,b)
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  x2=~x if dtype==torch.bool else x+1;b2=~b if dtype==torch.bool else b+2
  a.copy_(x2.npu());c.copy_(b2.npu());check(op(a,c).cpu(),x2,b2)
  assert torch.equal(a.cpu(),x2) and torch.equal(c.cpu(),b2)
  rows.append({'dtype':str(dtype),'n':n,'passed':True,'changed_inputs':True,'input_unchanged':True});print(rows[-1],flush=True)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 n=16387;k=module.AscendPowKernel((n,),(n,),dtype)
 padded=k.padded;x=torch.randn(n).to(dtype);b=torch.randn(n).to(dtype)
 xp=torch.nn.functional.pad(x,(0,padded-n));bp=torch.nn.functional.pad(b,(0,padded-n))
 explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
 guard=torch.full((padded+128,),17,dtype=dtype,device='npu');out=guard[64:64+padded];out.fill_(-19)
 explicit(xp.npu(),bp.npu(),out);torch.npu.synchronize()
 assert bool((guard[:64]==17).all()) and bool((guard[64+padded:]==17).all())
 actual=out.cpu();check(actual,xp,bp);bad=actual.clone();bad[n-1]=float('nan') if bool(torch.isfinite(actual[n-1])) else 0
 try:check(bad,xp,bp)
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'n':n,'allocated_elements':padded,'check':'sentinel_and_negative_control','passed':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('pow-tail-stress.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(Path(__file__).with_name('pow_reference.py').read_bytes()).hexdigest(),'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
