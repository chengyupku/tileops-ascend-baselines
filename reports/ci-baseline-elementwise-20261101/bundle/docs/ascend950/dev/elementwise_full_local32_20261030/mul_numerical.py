"""Mul full low-width patterns and wide random values across three operand patterns."""
import hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import arithmetic_mul as module
from tileops.ops.elementwise import MulFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
DTYPES=[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]
torch.manual_seed(42);rows=[];n=65537
for dtype in DTYPES:
 size=dtype.itemsize;storage={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[size]
 if size<=2:raw=torch.arange(n,dtype=torch.int64).to(storage)
 elif size==4:raw=torch.randint(-2147483648,2147483647,(n,),dtype=storage)
 else:raw=torch.randint(-2147483648,2147483647,(n,),dtype=storage)*4294967296+torch.randint(0,4294967295,(n,),dtype=storage)
 if dtype==torch.float32:
  raw[:8]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(dtype).tiny,1.,-1.],dtype=dtype).view(storage)
 x=raw.view(dtype);a=raw.npu().view(dtype);op=MulFwdOp()
 for mode in ['rolled','reversed','square']:
  other_raw=raw.roll(113) if mode=='rolled' else raw.flip(0) if mode=='reversed' else raw.clone()
  b=other_raw.view(dtype);c=other_raw.npu().view(dtype);actual=op(a,c).cpu()
  ref=(raw!=0)&(other_raw!=0) if dtype==torch.bool else torch.mul(x,b)
  compare_outputs(actual,ref,ElementwiseWorkload('MulFwdOp',(x,b)).verification(x,b))
  assert torch.equal(a.view(storage).cpu(),raw) and torch.equal(c.view(storage).cpu(),other_raw)
  rows.append({'dtype':str(dtype),'mode':mode,'values':n,'passed':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('mul-numerical.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
