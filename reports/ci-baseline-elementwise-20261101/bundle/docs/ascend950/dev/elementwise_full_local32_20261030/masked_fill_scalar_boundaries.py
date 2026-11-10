"""All dtype masked fill boundaries, bidirectional broadcasts, and runtime value changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import masked_fill_scalar as module
from tileops.ops.elementwise import MaskedFillScalarFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
DTYPES=[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float16,torch.bfloat16,torch.float32]
SHAPES=[((),()),((0,3),(1,3)),((1,),(1,)),((65,),(65,)),((129,),(1,)),((2,1,5),(1,3,1)),((1,5),(3,1)),((3,5),(1,5)),((2,3,0),(1,3,1)),((16387,),(16387,))]
rows=[]
def bits(x):return x.cpu().contiguous().reshape(-1).view(torch.uint8)
def draw(shape,dtype):
 if dtype==torch.bool:return torch.randint(0,2,shape).bool()
 if dtype.is_floating_point:return torch.randn(shape).to(dtype)
 return torch.randint(0 if dtype==torch.uint8 else -100,100,shape).to(dtype)
def verify(y,x,m,v):
 ref=x.masked_fill(m,v)
 compare_outputs(y,ref,ElementwiseWorkload('MaskedFillScalarFwdOp',(x,m),value=v.item()).verification(x,m))
for dtype in DTYPES:
 op=MaskedFillScalarFwdOp(value=0)
 for xs,ms in SHAPES:
  x=draw(xs,dtype);m=torch.randint(0,2,ms).bool();v=torch.tensor(False if dtype==torch.bool else 7,dtype=dtype)
  if x.ndim>=2 and x.numel():x=x.transpose(0,1).contiguous().transpose(0,1)
  a=x.npu();b=m.npu();op.value=v.item();before=[bits(t).clone() for t in (a,b)];out=op(a,b)
  assert out.shape==torch.broadcast_shapes(xs,ms) and out.dtype==dtype
  if out.numel():verify(out.cpu(),x,m,v)
  for t,snapshot in zip((a,b),before):assert torch.equal(bits(t),snapshot)
  v2=torch.tensor(True if dtype==torch.bool else 3,dtype=dtype);op.value=v2.item();b.logical_not_()
  second=op(a,b)
  if second.numel():verify(second.cpu(),x,~m,v2)
  assert torch.equal(bits(a),before[0])
  rows.append({'dtype':str(dtype),'input_shape':xs,'mask_shape':ms,'correct':True,'input_unchanged':True,'runtime_value_changed':True,'mask_changed':True});print(rows[-1],flush=True)
for dtype in DTYPES:
 n=16387;k=module.AscendMaskedFillScalarKernel(n,dtype,True if dtype==torch.bool else 7)
 x=draw((n,),dtype);m=torch.randint(0,2,(n,)).bool();v=torch.tensor(True if dtype==torch.bool else 7,dtype=dtype)
 a=x.npu();b=m.npu();k(a,b)
 explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
 guard=torch.full((n+128,),1,dtype=dtype,device='npu');out=guard[64:64+n];out.zero_()
 explicit(a.view(k.storage),b.view(torch.uint8),out.view(k.storage));torch.npu.synchronize()
 assert bool((guard[:64]==1).all()) and bool((guard[64+n:]==1).all())
 actual=out.cpu();verify(actual,x,m,v)
 bad=actual.clone();bad[-1]=~bad[-1] if dtype==torch.bool else bad[-1]+1
 try:verify(bad,x,m,v)
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'check':'sentinel_and_negative_control','correct':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('masked-fill-scalar-boundaries.json').write_text(json.dumps({'rows':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
