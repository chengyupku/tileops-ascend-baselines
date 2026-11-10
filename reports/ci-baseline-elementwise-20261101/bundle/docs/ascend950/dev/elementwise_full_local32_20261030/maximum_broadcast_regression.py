"""Non-power-of-two broadcast periods across vector tiles, with output guards."""
import hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import arithmetic_extrema as module
from tileops.ops.elementwise import MaximumFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from maximum_reference import maximum_reference
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for period,rows_count in [(p,n) for p in [960,1088,2112] for n in [5,6]]:
  x=torch.randn(rows_count,period).to(dtype);b=torch.randn(1,period).to(dtype);a=x.npu();c=b.npu()
  expected=maximum_reference(x,b);contract=ElementwiseWorkload('MaximumFwdOp',(x,b)).verification(x,b)
  compare_outputs(MaximumFwdOp()(a,c).cpu(),expected,contract)
  k=module.AscendMaximumKernel(x.shape,b.shape,dtype)
  assert k.period==period and not k.expand and k.padded==x.numel()
  explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
  n=x.numel();guard=torch.full((n+128,),17,dtype=dtype,device='npu');out=guard[64:64+n];out.fill_(-19)
  explicit(a.reshape(-1),c.reshape(-1),out);torch.npu.synchronize()
  assert bool((guard[:64]==17).all()) and bool((guard[64+n:]==17).all())
  compare_outputs(out.reshape_as(a).cpu(),expected,contract)
  assert torch.equal(a.cpu(),x) and torch.equal(c.cpu(),b)
  rows.append({'dtype':str(dtype),'period':period,'rows':rows_count,'passed':True,'sentinel':True,'input_unchanged':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('maximum-broadcast-regression.json').write_text(json.dumps({'reference_sha256':hashlib.sha256(Path(__file__).with_name('maximum_reference.py').read_bytes()).hexdigest(),'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'dependency_manifest':{str(Path(module.__file__).with_name('arithmetic.py')):hashlib.sha256(Path(module.__file__).with_name('arithmetic.py').read_bytes()).hexdigest()},'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
