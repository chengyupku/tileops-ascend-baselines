"""Non-power-of-two broadcast periods across vector tiles, with output guards."""
import hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import arithmetic as module
from tileops.ops.elementwise import AddFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for period in [960,1088,2112]:
  x=torch.randn(5,period).to(dtype);b=torch.randn(1,period).to(dtype);a=x.npu();c=b.npu();alpha=.5
  expected=torch.add(x,b,alpha=alpha);contract=ElementwiseWorkload('AddFwdOp',(x,b),alpha=alpha).verification(x,b)
  compare_outputs(AddFwdOp(alpha=alpha)(a,c).cpu(),expected,contract)
  k=module.AscendAddKernel(x.shape,b.shape,dtype,alpha)
  assert k.period==period and not k.expand and k.padded==x.numel()
  explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
  n=x.numel();guard=torch.full((n+128,),17,dtype=dtype,device='npu');out=guard[64:64+n];out.fill_(-19)
  explicit(a.reshape(-1),c.reshape(-1),out);torch.npu.synchronize()
  assert bool((guard[:64]==17).all()) and bool((guard[64+n:]==17).all())
  compare_outputs(out.reshape_as(a).cpu(),expected,contract)
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  rows.append({'dtype':str(dtype),'period':period,'rows':5,'alpha':alpha,'passed':True,'sentinel':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('add-broadcast-regression.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
