"""All dtype masked fill boundaries, bidirectional broadcasts, and runtime value changes."""
import json,hashlib
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import masked_fill_tensor as module
from tileops.ops.elementwise import MaskedFillTensorFwdOp
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
 compare_outputs(y,ref,ElementwiseWorkload('MaskedFillTensorFwdOp',(x,m,v)).verification(x,m,v))
for dtype in DTYPES:
 op=MaskedFillTensorFwdOp()
 for xs,ms in SHAPES:
  x=draw(xs,dtype);m=torch.randint(0,2,ms).bool();v=torch.tensor(False if dtype==torch.bool else 7,dtype=dtype)
  if x.ndim>=2 and x.numel():x=x.transpose(0,1).contiguous().transpose(0,1)
  a=x.npu();b=m.npu();c=v.npu();before=[bits(t).clone() for t in (a,b,c)];out=op(a,b,c)
  assert out.shape==torch.broadcast_shapes(xs,ms) and out.dtype==dtype
  if out.numel():verify(out.cpu(),x,m,v)
  for t,snapshot in zip((a,b,c),before):assert torch.equal(bits(t),snapshot)
  v2=torch.tensor(True if dtype==torch.bool else 3,dtype=dtype);c.copy_(v2.npu());b.logical_not_()
  second=op(a,b,c)
  if second.numel():verify(second.cpu(),x,~m,v2)
  assert torch.equal(bits(a),before[0])
  rows.append({'dtype':str(dtype),'input_shape':xs,'mask_shape':ms,'correct':True,'input_unchanged':True,'runtime_value_changed':True,'mask_changed':True});print(rows[-1],flush=True)
for dtype in DTYPES:
 n=16387;k=module.AscendMaskedFillTensorKernel(n,dtype)
 x=draw((n,),dtype);m=torch.randint(0,2,(n,)).bool();v=torch.tensor(True if dtype==torch.bool else 7,dtype=dtype)
 a=x.npu();b=m.npu();c=v.npu();k(a,b,c)
 explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
 guard=torch.full((n+128,),1,dtype=dtype,device='npu');out=guard[64:64+n];out.zero_()
 explicit(a.view(k.storage),b.view(torch.uint8),c.reshape(-1).view(k.storage),out.view(k.storage));torch.npu.synchronize()
 assert bool((guard[:64]==1).all()) and bool((guard[64+n:]==1).all())
 actual=out.cpu();verify(actual,x,m,v)
 bad=actual.clone();bad[-1]=~bad[-1] if dtype==torch.bool else bad[-1]+1
 try:verify(bad,x,m,v)
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'check':'sentinel_and_negative_control','correct':True});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('masked-fill-tensor-boundaries.json').write_text(json.dumps({'rows':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
