"""Hardtanh bound specialization, ties and IEEE values on small/tiled paths."""
import json,hashlib,struct,argparse
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import hardtanh as module
from tileops.ops.elementwise import HardtanhFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
rows=[];source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest()
for lo,hi in [(-1.,1.),(-.3,.7),(0.,1.),(-0.,1.),(-1.,0.),(-1.,-0.),(-0.,0.),(-2.,-1.),(1.,2.)]:
 op=HardtanhFwdOp(min_val=lo,max_val=hi)
 for n in [65,1048576]:
  if dtype!=torch.float32 and n>=65536:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).repeat(n//65536)
  else:
   f=torch.finfo(dtype);seed=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),f.max,-f.max,f.tiny,-f.tiny,f.tiny/2,-f.tiny/2,lo,hi,lo-.01,hi+.01],dtype=dtype);x=seed.repeat((n+len(seed)-1)//len(seed))[:n]
  device=x.npu();before=device.cpu().view(torch.uint8).clone();got=op(device).cpu();ref=torch.nn.functional.hardtanh(x,min_val=lo,max_val=hi)
  compare_outputs(got,ref,ElementwiseWorkload('HardtanhFwdOp',(device,),min_val=lo,max_val=hi).verification(device))
  zeros=(ref==0)&(got==0);assert torch.equal(torch.signbit(got[zeros]),torch.signbit(ref[zeros])),(lo,hi,n,'zero sign')
  assert torch.equal(before,device.cpu().view(torch.uint8))
  rows.append({'dtype':str(dtype),'min_bits':struct.pack('!d',lo).hex(),'max_bits':struct.pack('!d',hi).hex(),'n':n,'passed':True,'zero_sign':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
torch.npu.synchronize()
Path(__file__).with_name('hardtanh-parameters-'+a.dtype+'.json').write_text(json.dumps({'checks':rows,'module_sha256':frozen,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
