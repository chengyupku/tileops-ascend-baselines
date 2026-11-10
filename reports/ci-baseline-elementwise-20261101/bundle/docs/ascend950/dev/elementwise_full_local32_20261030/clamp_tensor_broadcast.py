import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import clamp_tensor as module
from tileops.ops.elementwise import ClampTensorFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest();rows=[];op=ClampTensorFwdOp()
cases=[((3,1),(1,5),()),((),(3,1),(1,5)),((2,1,7),(1,5,1),(7,)),((0,3),(1,3),()),((2,3,4,5),(1,3,1,1),(5,)),((1,1),(65,1),(1,3))]
def raw(t):return t.cpu().contiguous().reshape(-1).view(torch.uint8)
for shapes in cases:
 torch.manual_seed(317);cpu=[torch.randn(s).to(dtype) for s in shapes]
 cpu=[t.t().contiguous().t() if t.dim()==2 and t.numel() else t for t in cpu]
 allargs=[t.npu() for t in cpu]
 for mode in ['both','min','max']:
  args=(allargs[0],allargs[1] if mode!='max' else None,allargs[2] if mode!='min' else None)
  refs=(cpu[0],cpu[1] if mode!='max' else None,cpu[2] if mode!='min' else None)
  before=[raw(t).clone() if t is not None else None for t in args]
  ref=torch.clamp(*refs);out=op(*args);contract=ElementwiseWorkload('ClampTensorFwdOp',args).verification(*args)
  assert out.shape==ref.shape and out.dtype==dtype
  if out.numel():compare_outputs(out.cpu(),ref,contract)
  for v,b in zip(args,before):
   if v is not None:assert torch.equal(raw(v),b)
  changed=tuple((v+1) if v is not None else None for v in args)
  ref2=torch.clamp(*(v.cpu() if v is not None else None for v in changed));out2=op(*changed)
  if out2.numel():compare_outputs(out2.cpu(),ref2,contract)
  rows.append({'dtype':str(dtype),'shapes':shapes,'mode':mode,'correct':True,'input_unchanged':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
torch.npu.synchronize();Path(__file__).with_name('clamp-tensor-broadcast-'+a.dtype+'.json').write_text(json.dumps({'rows':rows,'module_sha256':digest,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
