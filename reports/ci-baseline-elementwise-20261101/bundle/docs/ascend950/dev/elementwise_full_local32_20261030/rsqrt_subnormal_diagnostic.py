import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import rsqrt
from tileops.ops.elementwise import RsqrtFwdOp
x=torch.tensor([-1.,0.,1e-38,1.],dtype=torch.float32).repeat(1024)
a=x.npu();y=RsqrtFwdOp()(a);e=torch.rsqrt(a);torch.npu.synchronize();ref=torch.rsqrt(x.double()).float()
torch.testing.assert_close(y.cpu(),ref,rtol=1e-5,atol=1e-5,equal_nan=True)
rows=[{'input':str(v.item()),'cpu_fp64':str(c.item()),'tileops':str(t.item()),'npu_eager':str(n.item())} for v,c,t,n in zip(x[:4],ref[:4],y.cpu()[:4],e.cpu()[:4])]
result={'rows':rows,'tileops_matches_cpu':True,'npu_eager_subnormal_is_positive_infinity':bool(torch.isposinf(e.cpu()[2])),'runtime_limit':torch.npu.get_device_limit(0)}
Path(__file__).with_name('rsqrt-subnormal-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
