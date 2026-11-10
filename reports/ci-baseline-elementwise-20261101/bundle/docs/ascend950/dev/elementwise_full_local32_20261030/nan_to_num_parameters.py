"""Original exact contract, parameter casting, non-cascading replacements and finite bits."""
import json,hashlib,struct,argparse
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import nan_to_num as module
from tileops.ops.elementwise import NanToNumFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
rows=[];source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest()
parameters=[(0.,None,None),(-0.,0.,-0.),(.3,.7,-.3),(float('inf'),-float('inf'),float('nan')),(float('nan'),float('nan'),float('nan')),(1e40,1e40,-1e40),(1e-50,1e-50,-1e-50),(0.,float('inf'),-float('inf')),(1e6,1e6,-1e6),(1.,None,-1.),(-1.,1.,None),(-0.,None,None)]
def bits(v):return None if v is None else struct.pack('!d',v).hex()
for nan,pos,neg in parameters:
 op=NanToNumFwdOp(nan=nan,posinf=pos,neginf=neg)
 for n in [65,1048576]:
  if dtype!=torch.float32 and n>=65536:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).repeat(n//65536)
  else:
   f=torch.finfo(dtype);seed=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),f.max,-f.max,f.tiny,-f.tiny,f.tiny/2,-f.tiny/2,-.3,.7],dtype=dtype);x=seed.repeat((n+len(seed)-1)//len(seed))[:n]
  device=x.npu();before=device.cpu().view(torch.uint8).clone();got=op(device).cpu();ref=torch.nan_to_num(x,nan=nan,posinf=pos,neginf=neg)
  compare_outputs(got,ref,ElementwiseWorkload('NanToNumFwdOp',(device,),nan=nan,posinf=pos,neginf=neg).verification(device))
  z=(ref==0)&(got==0);assert torch.equal(torch.signbit(got[z]),torch.signbit(ref[z])),(nan,pos,neg,n,'zero sign')
  finite=torch.isfinite(x);assert torch.equal(got[finite].view(torch.uint8),x[finite].view(torch.uint8)),(nan,pos,neg,n,'finite bits changed')
  assert torch.equal(before,device.cpu().view(torch.uint8))
  rows.append({'dtype':str(dtype),'nan_bits':bits(nan),'pos_bits':bits(pos),'neg_bits':bits(neg),'n':n,'passed':True,'zero_sign':True,'finite_bits':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
torch.npu.synchronize();Path(__file__).with_name('nan-to-num-parameters-'+a.dtype+'.json').write_text(json.dumps({'checks':rows,'module_sha256':frozen,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
