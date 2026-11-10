"""Full 16-bit encodings and randomized FP32 ReLU values, CPU reference."""
from pathlib import Path
import hashlib,json
import torch,torch_npu
from tileops.kernels.ascend.elementwise import relu as module
from tileops.ops.elementwise import ReluFwdOp
from tileops.ops._signature_codegen import CheckError
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[];torch.manual_seed(137)
for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64]:
 if not dtype.is_floating_point:
  info=torch.iinfo(dtype)
  x=torch.randint(info.min,info.max,(65536,),dtype=torch.int64).to(dtype)
  x[:5]=torch.tensor([info.min,info.max,0,1,-1 if info.min<0 else 2],dtype=dtype)
 elif dtype.itemsize==2:x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
 else:
  x=torch.randint(-2147483648,2147483647,(65536,),dtype=torch.int64).to(torch.int32).view(torch.float32)
  x[:8]=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),torch.finfo(dtype).tiny,torch.finfo(dtype).tiny*torch.finfo(dtype).eps,-1.])
 storage=torch.uint8
 for inplace in [False,True]:
  a=x.npu()
  if not dtype.is_floating_point:
   try:ReluFwdOp(inplace=inplace)(a)
   except CheckError as error:assert 'input dtype is outside' in str(error)
   else:raise AssertionError('unsupported dtype accepted')
   row=dict(dtype=str(dtype),inplace=inplace,passed=True,rejected=True);rows.append(row);print(row,flush=True)
   continue
  ref=torch.relu(x);out=ReluFwdOp(inplace=inplace)(a)
  got=out.cpu();compare_outputs(got,ref,ElementwiseWorkload('ReluFwdOp',(x,)).verification(x))
  assert bool(((got==ref)|(torch.isnan(got)&torch.isnan(ref))).all())
  zero=(ref==0);assert torch.equal(torch.signbit(got[zero]),torch.signbit(ref[zero]))
  assert (out is a)==inplace
  if not inplace:assert torch.equal(a.cpu().view(storage),x.view(storage))
  row=dict(dtype=str(dtype),inplace=inplace,values=x.numel(),passed=True,zero_sign=True);rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('relu-numerical.json').write_text(json.dumps(dict(checks=rows,runtime_limit=torch.npu.get_device_limit(0),module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
