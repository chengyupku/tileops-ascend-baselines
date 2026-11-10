import json,math
from pathlib import Path
import torch,torch_npu,tilelang
import tilelang.ascend.language as T
from tileops.kernels.ascend.elementwise import sinusoidal as module
from workloads.elementwise import sinusoidal_reference
r=Path(__file__).resolve().parent
D=256;half=D//2
@T.prim_func
def divs(Y:T.Tensor((half,), 'float32')):
 with T.Kernel(1):
  v=T.alloc_shared((half,),'float32')
  module._divisors(v,half,0,D)
  T.copy(v,Y)
k=tilelang.compile(divs,target='ascend',out_idx=[0]);gotdiv=k().cpu()
e=torch.arange(0,D,2,dtype=torch.float32)/D;cpu=torch.pow(10000.,e);npu=torch.pow(10000.,e.npu()).cpu();host=torch.tensor([math.pow(10000.,float(x)) for x in e],dtype=torch.float32)
op=module.AscendSinusoidalKernel(512,D,torch.float32,'npu');got=op().cpu();cpuref=sinusoidal_reference(512,D,torch.float32,'cpu');npuref=sinusoidal_reference(512,D,torch.float32,'npu').cpu()
def diff(a,b):
 bad=(a-b).abs()>1e-5+1e-5*b.abs()
 return dict(max_abs=float((a-b).abs().max()),mismatch=int(bad.sum()))
report=dict(div_kernel_cpu=diff(gotdiv,cpu),div_kernel_npu=diff(gotdiv,npu),div_npu_cpu=diff(npu,cpu),host_cpu=diff(host,cpu),output_cpu=diff(got,cpuref),output_npu=diff(got,npuref),divs=[dict(i=i,k=float(gotdiv[i]),cpu=float(cpu[i]),npu=float(npu[i]),host=float(host[i])) for i in range(16)],runtime_limit=torch.npu.get_device_limit(0))
(r/'sinusoidal-probe.json').write_text(json.dumps(report,indent=2)+'\n');(r/'sinusoidal-probe-div.c').write_text(k.get_kernel_source());(r/'sinusoidal-probe-output.c').write_text(op.compiled.get_kernel_source());print(json.dumps(report,indent=2));torch.npu.synchronize()
