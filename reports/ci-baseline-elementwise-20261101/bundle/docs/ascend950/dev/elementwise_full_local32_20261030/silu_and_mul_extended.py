"""CPU contract checks for gated rows, dtype, empty and partial tiles, input/output guards."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import silu_and_mul as module
from tileops.ops.elementwise import SiluAndMulFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest();op=SiluAndMulFwdOp();rows=[]
shapes=[(0,7),(3,0),(1,1),(3,7),(1,63),(2,64),(3,65),(5,129),(2,16387),(65,129),(1,65536),(32,769)]
for m,n in shapes:
 torch.manual_seed(1729);x=(torch.randn((m,2*n))*3).to(dtype)
 if n==65536:
  if dtype!=torch.float32:x[0,:n]=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
  else:x[0,:15]=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),1e-40,-1e-40,88.,-88.,90.,-90.,1e38,-1e38,1.,-1.])
  x[0,n:]=1
 if m==3 and n==7:x=x.t().contiguous().t()
 device=x.npu();before=device.cpu().contiguous().view(torch.uint8).clone()
 workload=ElementwiseWorkload('SiluAndMulFwdOp',(device,));contract=workload.verification(device);ref=workload.ref_program(x)
 out=op(device);torch.npu.synchronize();assert out.shape==(m,n) and out.dtype==dtype
 compare_outputs(out.cpu(),ref,contract);assert torch.equal(before,device.cpu().contiguous().view(torch.uint8))
 if x.numel():assert out.untyped_storage().data_ptr()!=device.untyped_storage().data_ptr()
 if n in [65,16387,769]:
  changed=(x*.5+1).npu();compare_outputs(op(changed).cpu(),workload.ref_program(changed.cpu()),contract)
 if n==16387:
  built=[k for entries in op._built_entries.values() for k in entries.values() if k.m==m and k.n==n and k.dtype==dtype];assert len(built)==1
  explicit=tilelang.compile(built[0].kernel.prim_func,target='ascend',out_idx=[])
  guard=torch.full((m*n+128,),321,device='npu',dtype=dtype);target=guard[64:-64].reshape(m,n);explicit(device,target);torch.npu.synchronize()
  assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());compare_outputs(target.cpu(),ref,contract)
  bad=target.cpu();bad[-1,-1]=float('nan')
  try:compare_outputs(bad,ref,contract)
  except AssertionError:pass
  else:raise AssertionError('checker accepted corrupted tail')
 rows.append(dict(dtype=str(dtype),m=m,n=n,correct=True,input_unchanged=True));print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen;torch.npu.synchronize()
Path(__file__).with_name('silu-and-mul-extended-'+a.dtype+'.json').write_text(json.dumps(dict(rows=rows,module_sha256=frozen,runtime_limit=torch.npu.get_device_limit(0)),indent=2)+'\n')
