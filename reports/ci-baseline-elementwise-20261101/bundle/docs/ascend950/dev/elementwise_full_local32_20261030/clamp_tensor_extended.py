"""Original exact contract, full bit domains, noncontiguous inputs and guarded tails."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import clamp_tensor as module
from tileops.ops.elementwise import ClampTensorFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest();rows=[];op=ClampTensorFwdOp()
def raw(t):return t.cpu().contiguous().reshape(-1).view(torch.uint8)
for shape in [(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)]:
 torch.manual_seed(1729);cpu=[(torch.randn(shape)*3).to(dtype) for _ in range(3)]
 if shape==(65536,):
  if dtype!=torch.float32:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype);cpu=[x,x.roll(12345),x.roll(4321)]
  else:
   special=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1e-40,-1e-40,1e-38,-1e-38,1e38,-1e38])
   for i,t in enumerate(cpu):t[:len(special)]=special.roll(i)
 if len(shape)==2 and cpu[0].numel():cpu=[t.t().contiguous().t() for t in cpu]
 allargs=[t.npu() for t in cpu]
 for mode in ['both','min','max']:
  args=(allargs[0],allargs[1] if mode!='max' else None,allargs[2] if mode!='min' else None)
  refs=(cpu[0],cpu[1] if mode!='max' else None,cpu[2] if mode!='min' else None)
  before=[raw(t).clone() if t is not None else None for t in args]
  contract=ElementwiseWorkload('ClampTensorFwdOp',args).verification(*args);ref=torch.clamp(*refs);out=op(*args);torch.npu.synchronize()
  assert out.shape==ref.shape and out.dtype==dtype
  if out.numel():
   compare_outputs(out.cpu(),ref,contract)
   assert all(out.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in args if v is not None)
  for v,b in zip(args,before):
   if v is not None:assert torch.equal(raw(v),b)
  if shape in [(32767,),(524289,),(540671,)]:
   changed=tuple((v.abs()+1) if v is not None else None for v in args)
   compare_outputs(op(*changed).cpu(),torch.clamp(*(v.cpu() if v is not None else None for v in changed)),contract)
  if shape==(16387,):
   built=[k for e in op._built_entries.values() for k in e.values() if k.n_total==16387 and k.has_min==(mode!='max') and k.has_max==(mode!='min')]
   assert len(built)==1
   explicit=tilelang.compile(built[0].compiled.prim_func,target='ascend',out_idx=[])
   guard=torch.full((16387+128,),321,device='npu',dtype=dtype);output=guard[64:-64]
   explicit(args[0],args[0] if args[1] is None else args[1],args[0] if args[2] is None else args[2],output);torch.npu.synchronize()
   assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());compare_outputs(output.cpu(),ref,contract)
   bad=output.cpu();bad[-1]=float('nan')
   try:compare_outputs(bad,ref,contract)
   except AssertionError:pass
   else:raise AssertionError('checker accepted a corrupt tail')
  rows.append({'dtype':str(dtype),'shape':shape,'mode':mode,'correct':True,'input_unchanged':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
torch.npu.synchronize();Path(__file__).with_name('clamp-tensor-extended-'+a.dtype+'.json').write_text(json.dumps({'rows':rows,'module_sha256':digest,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
