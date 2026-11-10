"""Sub scalar alpha extremes retain integer wraparound semantics."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.ops.elementwise import SubFwdOp
from tileops.kernels.ascend.elementwise import arithmetic_sub as module
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
for dtype in [torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64]:
 info=torch.iinfo(dtype);values=[0,1,2,info.min,info.max];x=torch.tensor(values,dtype=dtype).repeat(26)[:129];b=x.roll(3)
 a=x.npu();c=b.npu();op=SubFwdOp()
 alphas=[0,1,2,255] if dtype==torch.uint8 else [0,1,-1,info.min,info.max]
 for alpha in alphas:
  # Exact integer subtraction in the declared output dtype, using Python
  # integers so the oracle cannot overflow before the explicit wrap.
  bits=dtype.itemsize*8;modulus=1<<bits;offset=0 if dtype==torch.uint8 else 1<<(bits-1)
  ref=torch.tensor([((int(v)-alpha*int(w)+offset)%modulus)-offset for v,w in zip(x,b)],dtype=dtype)
  op.alpha=alpha
  row={'dtype':str(dtype),'alpha':alpha,'values':129,'passed':False}
  try:
   cpu=torch.sub(x,b,alpha=alpha)
  except RuntimeError as e:row['cpu_reference_error']=str(e)
  else:
   assert torch.equal(cpu,ref),'Independent modular oracle differs from CPU torch.sub'
   row['cpu_reference_agrees']=True
  try:
   got=op(a,c).cpu();compare_outputs(got,ref,ElementwiseWorkload('SubFwdOp',(x,b),alpha=alpha).verification(x,b))
   assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
   row.update(passed=True,input_unchanged=True)
  except Exception as e:row['error']=str(e)
  rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('sub-alpha-extremes.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
assert all(v['passed'] for v in rows),'Sub alpha extreme failure'
