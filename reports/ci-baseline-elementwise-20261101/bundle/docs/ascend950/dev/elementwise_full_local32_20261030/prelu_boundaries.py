"""PReLU shape, scalar weight, large-channel gather, replay, and output bounds."""
import hashlib,json
from pathlib import Path
import torch,torch_npu,tilelang
import torch.nn.functional as F
from tileops.kernels.ascend.elementwise import prelu as module
from tileops.ops.elementwise import PreluFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
def bits(x):return x.cpu().contiguous().reshape(-1).view(torch.uint8)
def check(y,x,w):
 ref=F.prelu(x.double(),w.double()).to(x.dtype)
 compare_outputs(y,ref,ElementwiseWorkload('PreluFwdOp',(x,w)).verification(x,w))
shapes=[((0,3,5),1),((),0),((1,),1),((63,),1),((65,),1),((129,),1),((3,5),5),((2,3,5),3),((2,3,4097),3),((2,8193,1),8193),((2,8192,1),8192),((2,3,5),1),((2,3,0),3),((1,3,65),3),((16387,),1)]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 op=PreluFwdOp()
 for shape,c in shapes:
  x=torch.randn(shape).to(dtype);w=torch.randn((c,) if c else ()).to(dtype)
  if len(shape)>=2 and x.numel():x=x.transpose(0,1).contiguous().transpose(0,1)
  if c>1:
   backing=torch.zeros(c*2,dtype=dtype);backing[::2]=w;w=backing[::2]
  a=x.npu();b=w.npu();before=(bits(a).clone(),bits(b).clone());out=op(a,b)
  assert out.shape==x.shape and out.dtype==dtype
  if x.numel():
   check(out.cpu(),x,w)
   assert out.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
   changed=w.neg()+.25;check(op(a,changed.npu()).cpu(),x,changed)
  assert torch.equal(bits(a),before[0]) and torch.equal(bits(b),before[1])
  rows.append({'dtype':str(dtype),'shape':shape,'weight_count':c,'correct':True,'input_unchanged':True,'changed_weight':True});print(rows[-1],flush=True)
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 shape=(1,3,65);n=195;k=module.AscendPreluKernel(n,dtype,3,65)
 explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
 x=torch.randn(shape).to(dtype);w=torch.tensor([.1,-.25,1.5],dtype=dtype)
 a=x.npu();b=w.npu();guarded=torch.full((n+128,),321,dtype=dtype,device='npu');out=guarded[64:64+n];out.zero_()
 explicit(a.reshape(-1),b,out);torch.npu.synchronize()
 assert bool((guarded[:64]==321).all()) and bool((guarded[64+n:]==321).all())
 actual=out.cpu().reshape(shape);check(actual,x,w)
 bad=actual.clone();bad.flatten()[-1]=float('nan')
 try:check(bad,x,w)
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'check':'sentinel_and_negative_control','correct':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('prelu-boundaries.json').write_text(json.dumps({'rows':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
