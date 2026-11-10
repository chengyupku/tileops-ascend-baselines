"""Minimum full low-width patterns and wide random values across three operand patterns."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_extrema as module
from tileops.ops.elementwise import MinimumFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from minimum_reference import minimum_reference
DTYPES=[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]
torch.manual_seed(42);rows=[];n=65537
for dtype in DTYPES:
 size=dtype.itemsize;storage={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[size]
 if size<=2:raw=torch.arange(n,dtype=torch.int64).to(storage)
 elif size==4:raw=torch.randint(-2147483648,2147483647,(n,),dtype=storage)
 else:raw=torch.randint(-2147483648,2147483647,(n,),dtype=storage)*4294967296+torch.randint(0,4294967295,(n,),dtype=storage)
 if dtype==torch.float32:
  raw[:8]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(dtype).tiny,1.,-1.],dtype=dtype).view(storage)
 x=raw.view(dtype);a=raw.npu().view(dtype);op=MinimumFwdOp()
 for mode in ['rolled','reversed','square']:
  other_raw=raw.roll(113) if mode=='rolled' else raw.flip(0) if mode=='reversed' else raw.clone()
  b=other_raw.view(dtype);c=other_raw.npu().view(dtype);actual=op(a,c).cpu()
  ref=minimum_reference(x,b)
  row={'dtype':str(dtype),'mode':mode,'values':n,'passed':True}
  try:
   compare_outputs(actual,ref,ElementwiseWorkload('MinimumFwdOp',(x,b)).verification(x,b))
   if dtype.is_floating_point:
    zero=(actual==0)&(ref==0)
    assert torch.equal(actual.signbit()[zero],ref.signbit()[zero]), 'signed-zero mismatch'
  except AssertionError as exc:
   row.update(passed=False,error=str(exc))
   bad=actual!=ref
   if dtype.is_floating_point:
    bad=(bad & ~(actual.isnan()&ref.isnan())) | ((actual==0)&(ref==0)&(actual.signbit()!=ref.signbit()))
   ids=bad.nonzero().flatten()[:12].tolist()
   row['examples']=[{'index':i,'a':x[i].item(),'b':b[i].item(),'actual':actual[i].item(),'expected':ref[i].item()} for i in ids]
  assert torch.equal(a.view(storage).cpu(),raw) and torch.equal(c.view(storage).cpu(),other_raw)
  row['input_unchanged']=True;rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('minimum-numerical.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(__file__).with_name('minimum_reference.py').read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name(name)):hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest() for name in ['arithmetic.py','_bitwise_fast.py','_bitwise_fused.py']},'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
assert all(v['passed'] for v in rows)
