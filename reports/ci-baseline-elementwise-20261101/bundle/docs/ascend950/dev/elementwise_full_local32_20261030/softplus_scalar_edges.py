"""Slope specialization, IEEE values and zero-sign cache separation on both paths."""
import json,hashlib,struct,argparse
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import softplus as module
from tileops.ops.elementwise import SoftplusFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for beta,threshold in [(0.,-1.),(-0.,-1.),(1e-50,20.),(-1e-50,20.),(-float('inf'),20.),(1e40,20.),(1.,-float('inf')),(1.,-0.)]:
  op=SoftplusFwdOp(beta=beta,threshold=threshold)
  for n in [65]:
   if dtype!=torch.float32 and n>=65536:
    x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype).repeat(n//65536)
   else:
    f=torch.finfo(dtype);seed=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),f.max,-f.max,f.tiny,-f.tiny,f.tiny/2,-f.tiny/2,-1.,1.,-0.001,0.001],dtype=dtype);x=seed.repeat((n+len(seed)-1)//len(seed))[:n]
   device=x.npu();before=device.cpu().view(torch.uint8).clone()
   try:ref=torch.nn.functional.softplus(x,beta=beta,threshold=threshold)
   except RuntimeError as error:
    assert 'overflow' in str(error)
    try:op(device)
    except RuntimeError as actual:assert 'overflow' in str(actual)
    else:raise AssertionError('kernel accepted a scalar rejected by the CPU reference')
    assert torch.equal(before,device.cpu().view(torch.uint8))
    rows.append({'dtype':str(dtype),'beta_bits':struct.pack('!d',beta).hex(),'threshold_bits':struct.pack('!d',threshold).hex(),'n':n,'passed':True,'zero_sign':True,'rejected_overflow':True});print(rows[-1],flush=True)
    continue
   got=op(device).cpu()
   compare_outputs(got,ref,ElementwiseWorkload('SoftplusFwdOp',(device,),beta=beta,threshold=threshold).verification(device))
   zeros=(ref==0)&(got==0);assert torch.equal(torch.signbit(got[zeros]),torch.signbit(ref[zeros])),(beta,threshold,n,'zero sign')
   assert torch.equal(before,device.cpu().view(torch.uint8))
   rows.append({'dtype':str(dtype),'beta_bits':struct.pack('!d',beta).hex(),'threshold_bits':struct.pack('!d',threshold).hex(),'n':n,'passed':True,'zero_sign':True});print(rows[-1],flush=True)

assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
torch.npu.synchronize()
Path(__file__).with_name('softplus-scalar-edges.json').write_text(json.dumps({'checks':rows,'module_sha256':frozen,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
