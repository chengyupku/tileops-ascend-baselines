"""Full low-width patterns and wide random FloorDivide pairs under original exact contract."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
import floor_divide_widen_candidate as module
from tileops.ops.elementwise import FloorDivideFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from floor_divide_reference import floor_divide_reference
rows=[];torch.manual_seed(42);n=65537
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 storage=torch.int16 if dtype.itemsize==2 else torch.int32
 raw=torch.arange(n,dtype=torch.int64).to(storage) if dtype.itemsize==2 else torch.randint(-2147483648,2147483647,(n,),dtype=storage)
 if dtype==torch.float32:
  raw[:12]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(dtype).tiny,1.,-1.,1e-38,-1e-38,1e38,-1e38],dtype=dtype).view(storage)
 x=raw.view(dtype);a=raw.npu().view(dtype);op=module.FloorDivideImplementation((n,),(n,),dtype,rounding_mode="floor")
 for pattern in ['rolled','reversed','square']:
  other_raw=raw.roll(113) if pattern=='rolled' else raw.flip(0) if pattern=='reversed' else raw.clone()
  b=other_raw.view(dtype);c=other_raw.npu().view(dtype);actual=op(a,c).cpu();ref=floor_divide_reference(x,b)
  cpu=torch.floor_divide(x,b);cpu_bad=~((cpu==ref)|(cpu.isnan()&ref.isnan()))
  row={'dtype':str(dtype),'pattern':pattern,'values':n,'passed':True,'cpu_reference_mismatches':int(cpu_bad.sum())}
  try:compare_outputs(actual,ref,ElementwiseWorkload('FloorDivideFwdOp',(x,b)).verification(x,b))
  except AssertionError as e:
   row['passed']=False;row['error']=str(e);bad=~((actual==ref)|(actual.isnan()&ref.isnan()));ids=bad.nonzero().flatten()[:12].tolist();row['examples']=[{'index':i,'a':x[i].item(),'b':b[i].item(),'actual':actual[i].item(),'expected':ref[i].item()} for i in ids]
  assert torch.equal(a.view(storage).cpu(),raw) and torch.equal(c.view(storage).cpu(),other_raw)
  row['input_unchanged']=True;rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('floor-divide-widen-numerical.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(Path(__file__).with_name('floor_divide_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
assert all(v['passed'] for v in rows)
