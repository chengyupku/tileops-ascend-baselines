"""Full low-width patterns and wide random LerpScalar pairs under original workload contract."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import lerp as module
from tileops.ops.elementwise import LerpScalarFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from lerp_scalar_reference import lerp_scalar_reference
rows=[];torch.manual_seed(42);n=65537
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 storage=torch.int16 if dtype.itemsize==2 else torch.int32
 raw=torch.arange(n,dtype=torch.int64).to(storage) if dtype.itemsize==2 else torch.randint(-2147483648,2147483647,(n,),dtype=storage)
 if dtype==torch.float32:
  raw[:12]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(dtype).tiny,1.,-1.,1e-38,-1e-38,1e38,-1e38],dtype=dtype).view(storage)
 x=raw.view(dtype);a=raw.npu().view(dtype);op=LerpScalarFwdOp()
 patterns=[('rolled',raw.roll(113)),('reversed',raw.flip(0)),('square',raw.clone())]
 for pattern,other_raw in patterns:
  b=other_raw.view(dtype);c=other_raw.npu().view(dtype);actual=op(a,c).cpu();ref=lerp_scalar_reference(x,b)
  row={'dtype':str(dtype),'pattern':pattern,'values':n,'passed':True}
  try:
   compare_outputs(actual,ref,ElementwiseWorkload('LerpScalarFwdOp',(x,b)).verification(x,b))
   zero=(actual==0)&(ref==0)
   assert torch.equal(actual.signbit()[zero],ref.signbit()[zero]), 'signed zero mismatch'
  except AssertionError as e:
   row['passed']=False;row['error']=str(e);bad=~((actual==ref)|(actual.isnan()&ref.isnan()));ids=bad.nonzero().flatten()[:12].tolist();row['examples']=[{'index':i,'a':x[i].item(),'b':b[i].item(),'actual':actual[i].item(),'expected':ref[i].item()} for i in ids]
  assert torch.equal(a.view(storage).cpu(),raw) and torch.equal(c.view(storage).cpu(),other_raw)
  row['input_unchanged']=True;rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('lerp_scalar-numerical.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name('arithmetic.py')):hashlib.sha256(Path(module.__file__).with_name('arithmetic.py').read_bytes()).hexdigest()},'reference_sha256':hashlib.sha256(Path(__file__).with_name('lerp_scalar_reference.py').read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
assert all(v['passed'] for v in rows)
