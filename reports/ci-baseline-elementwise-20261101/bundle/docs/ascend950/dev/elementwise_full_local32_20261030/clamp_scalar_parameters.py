"""Scalar bound combinations under the original zero-tolerance contract."""
import json,hashlib,struct,argparse
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import clamp_scalar as module
from tileops.ops.elementwise import ClampScalarFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
rows=[];source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest();tiny=torch.finfo(dtype).tiny/2
bounds=[(-.5,.5),(-.3,.7),(None,0.),(0.,None),(-0.,None),(None,-0.),(-0.,0.),(0.,-0.),(1.,-1.),(float('nan'),1.),(-1.,float('nan')),(None,float('nan')),(float('nan'),None),(float('inf'),None),(None,-float('inf')),(-float('inf'),float('inf')),(None,.7),(-.3,None),(tiny,None),(None,-tiny)]
def bits(v):return None if v is None else struct.pack('!d',v).hex()
for lo,hi in bounds:
 op=ClampScalarFwdOp(min=lo,max=hi)
 for n in [65,1048576]:
  if dtype!=torch.float32 and n>=65536:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).repeat(n//65536)
  else:
   f=torch.finfo(dtype);values=[0.,-0.,float('inf'),-float('inf'),float('nan'),f.max,-f.max,f.tiny,-f.tiny,f.tiny/2,-f.tiny/2,-.3,.7]
   values += [v for v in (lo,hi) if v is not None]
   seed=torch.tensor(values,dtype=dtype);x=seed.repeat((n+len(seed)-1)//len(seed))[:n]
  device=x.npu();before=device.cpu().view(torch.uint8).clone();got=op(device).cpu();ref=torch.clamp(x,min=lo,max=hi)
  compare_outputs(got,ref,ElementwiseWorkload('ClampScalarFwdOp',(device,),min=lo,max=hi).verification(device))
  z=(ref==0)&(got==0);assert torch.equal(torch.signbit(got[z]),torch.signbit(ref[z])),(lo,hi,n,'zero sign')
  assert torch.equal(before,device.cpu().view(torch.uint8))
  rows.append({'dtype':str(dtype),'min_bits':bits(lo),'max_bits':bits(hi),'n':n,'passed':True,'zero_sign':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
torch.npu.synchronize();Path(__file__).with_name('clamp-scalar-parameters-'+a.dtype+'.json').write_text(json.dumps({'checks':rows,'module_sha256':frozen,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
