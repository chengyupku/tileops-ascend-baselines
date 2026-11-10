"""Lerp full-input CPU reference, broadcast, rounding boundaries and guarded tails."""
import argparse,hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import lerp as module
from tileops.ops.elementwise import LerpTensorFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest();rows=[];op=LerpTensorFwdOp()
def raw(t):return t.cpu().contiguous().reshape(-1).view(torch.uint8)
shapes=[(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)]
broadcast=[((1,7),(5,1),()),((),(3,5),(1,5)),((2,1,5),(1,3,1),(2,3,5)),((3,1),(1,5),(3,5)),((0,1),(1,5),()),((1,),(129,),(1,)),((17,1),(1,769),())]
for xs,zs,ws in [(s,s,s) for s in shapes]+broadcast:
 torch.manual_seed(1729)
 x=(torch.randn(xs)*3).to(dtype);z=(torch.randn(zs)*3).to(dtype);w=torch.rand(ws).to(dtype)
 if xs==zs==ws==(65536,):
  if dtype!=torch.float32:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype);z=x.roll(12345)
  else:
   special=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1e-40,-1e-40,1e-38,-1e-38,1e38,-1e38])
   x[:len(special)]=special;z[:len(special)]=special.roll(1)
  weights=torch.tensor([-2.,-1.,-.5,-0.,0.,.25,.499,.5,.501,.75,1.,2.],dtype=dtype)
  w=weights.repeat((65536+len(weights)-1)//len(weights))[:65536]
 cpu=[x,z,w]
 if len(xs)==2 and all(t.ndim==2 for t in cpu):cpu=[t.t().contiguous().t() for t in cpu]
 args=[t.npu() for t in cpu];before=[raw(t).clone() for t in args];ref=torch.lerp(*cpu)
 contract=ElementwiseWorkload('LerpTensorFwdOp',args).verification(*args);out=op(*args);torch.npu.synchronize()
 assert out.shape==ref.shape and out.dtype==dtype
 if out.numel():
  compare_outputs(out.cpu(),ref,contract)
  assert all(out.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in args)
 assert all(torch.equal(raw(t),b) for t,b in zip(args,before))
 if xs in [(32767,),(524289,),(540671,)]:
  changed=(args[1],args[0],1-args[2]);got=op(*changed);expected=torch.lerp(*(v.cpu() for v in changed));compare_outputs(got.cpu(),expected,contract)
 if xs==(16387,):
  built=[k for e in op._built_entries.values() for k in e.values() if k.n==16387];assert len(built)==1
  kernel=built[0];explicit=tilelang.compile(kernel.compiled.prim_func,target='ascend',out_idx=[])
  guard=torch.full((kernel.padded+128,),321,device='npu',dtype=dtype);output=guard[64:-64]
  padded=[torch.nn.functional.pad(v,(0,kernel.padded-v.numel())) for v in args]
  explicit(*padded,output);torch.npu.synchronize()
  assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());compare_outputs(output[:16387].cpu(),ref,contract)
  bad=output[:16387].cpu();bad[-1]=float('nan')
  try:compare_outputs(bad,ref,contract)
  except AssertionError:pass
  else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'input_shape':xs,'end_shape':zs,'weight_shape':ws,'correct':True,'input_unchanged':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
torch.npu.synchronize();Path(__file__).with_name('lerp-tensor-extended-'+a.dtype+'.json').write_text(json.dumps({'rows':rows,'module_sha256':digest,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
