"""Full bit selection, general broadcasting, changed inputs and guarded output."""
import argparse,hashlib,json,math
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import where as module
from tileops.ops.elementwise import WhereFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
p=argparse.ArgumentParser();p.add_argument('--dtype',required=True);a=p.parse_args();dtype=getattr(torch,a.dtype)
source=Path(module.__file__);digest=hashlib.sha256(source.read_bytes()).hexdigest();rows=[];op=WhereFwdOp()
def raw(t):return t.cpu().contiguous().reshape(-1).view(torch.uint8)
shapes=[(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)]
broadcast=[((1,7),(5,1),()),((),(3,5),(1,5)),((2,1,5),(1,3,1),(2,3,5)),((3,1),(1,5),(3,5)),((0,1),(1,5),()),((1,),(129,),(1,)),((17,1),(1,769),())]
for cs,xs,zs in [(s,s,s) for s in shapes]+broadcast:
 torch.manual_seed(1729)
 c=torch.randint(0,2,cs).bool();x=(torch.randn(xs)*3).to(dtype);z=(torch.randn(zs)*3).to(dtype)
 if xs==zs==(65536,):
  if dtype!=torch.float32:
   x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype);z=x.roll(12345)
  else:
   special=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1e-40,-1e-40,1e-38,-1e-38,1e38,-1e38])
   x[:len(special)]=special;z[:len(special)]=special.roll(1)
 cpu=[c,x,z]
 if len(xs)==2 and all(t.ndim==2 for t in cpu):cpu=[t.t().contiguous().t() for t in cpu]
 args=[t.npu() for t in cpu];before=[raw(t).clone() for t in args];ref=torch.where(*cpu)
 contract=ElementwiseWorkload('WhereFwdOp',args).verification(*args);out=op(*args);torch.npu.synchronize()
 assert out.shape==ref.shape and out.dtype==dtype
 if out.numel():
  compare_outputs(out.cpu(),ref,contract);assert torch.equal(raw(out),raw(ref)), 'selected bits changed'
  assert all(out.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in args)
 assert all(torch.equal(raw(t),b) for t,b in zip(args,before))
 if xs in [(32767,),(524289,),(540671,)]:
  changed=(~args[0],args[2],args[1]);got=op(*changed);expected=torch.where(*(v.cpu() for v in changed));assert torch.equal(raw(got),raw(expected))
 if xs==(16387,):
  built=[k for e in op._built_entries.values() for k in e.values() if k.n_total==16387];assert len(built)==1
  explicit=tilelang.compile(built[0].compiled.prim_func,target='ascend',out_idx=[])
  storage=built[0].storage;guard=torch.full((16387+128,),321,device='npu',dtype=dtype);output=guard[64:-64]
  explicit(args[0].view(torch.uint8),args[1].view(storage),args[2].view(storage),output.view(storage));torch.npu.synchronize()
  assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all());assert torch.equal(raw(output),raw(ref))
  bad=output.cpu();bad[-1]=float('nan')
  try:compare_outputs(bad,ref,contract)
  except AssertionError:pass
  else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'condition_shape':cs,'input_shape':xs,'other_shape':zs,'correct':True,'input_unchanged':True,'selected_bits_exact':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
torch.npu.synchronize();Path(__file__).with_name('where-extended-'+a.dtype+'.json').write_text(json.dumps({'rows':rows,'module_sha256':digest,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
