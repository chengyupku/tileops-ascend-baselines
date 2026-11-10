"""Default NanToNum replacements: CPU reference, full 16-bit domain, shapes and guarded tails."""
import json,hashlib
from pathlib import Path
import torch,torch_npu,tilelang
from tileops.kernels.ascend.elementwise import nan_to_num as module
from tileops.ops.elementwise import NanToNumFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
source=Path(module.__file__);frozen=hashlib.sha256(source.read_bytes()).hexdigest()
for mode in ['default']:
 op=NanToNumFwdOp()
 for dtype in [torch.float16,torch.bfloat16,torch.float32]:
  for shape in [(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,),(32767,),(524289,),(540671,),(65536,)]:
   torch.manual_seed(1729)
   x=(torch.randn(shape)*3).to(dtype)
   if shape==(65536,):
    if dtype!=torch.float32:x=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(dtype)
    else:
     x[:13]=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1e-40,-1e-40,1e-38,-1e-38,1e38,-1e38,8.,-8.])
   if len(shape)==2 and x.numel():x=x.t().contiguous().t()
   a=x.npu();before=a.cpu().contiguous().view(torch.uint8).clone() if a.dim() else a.cpu().reshape(1).view(torch.uint8).clone()
   ref=torch.nan_to_num(x);out=op(a);torch.npu.synchronize()
   assert out.shape==x.shape and out.dtype==x.dtype
   contract=ElementwiseWorkload('NanToNumFwdOp',(a,)).verification(a)
   if x.numel():
    compare_outputs(out.cpu(),ref,contract)
    assert out.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
   after=a.cpu().contiguous().view(torch.uint8) if a.dim() else a.cpu().reshape(1).view(torch.uint8)
   assert torch.equal(before,after)
   if shape in [(32767,),(524289,),(540671,)]:
    changed=(x.abs()+1).npu();compare_outputs(op(changed).cpu(),torch.nan_to_num(changed.cpu()),contract)
   if shape==(16387,):
    built=[k for entries in op._built_entries.values() for k in entries.values() if k.n_total==16387 and k.dtype==dtype]
    assert len(built)==1
    explicit=tilelang.compile(built[0].compiled.prim_func,target='ascend',out_idx=[])
    guard=torch.full((x.numel()+128,),321,device='npu',dtype=dtype);output=guard[64:-64];explicit(a,output);torch.npu.synchronize()
    assert bool((guard[:64]==321).all()) and bool((guard[-64:]==321).all())
    compare_outputs(output.cpu(),ref,contract)
    bad=output.cpu();bad[-1]=float('nan')
    try:compare_outputs(bad,ref,contract)
    except AssertionError:pass
    else:raise AssertionError('checker accepted corrupt tail')
   rows.append({'mode':mode,'dtype':str(dtype),'shape':shape,'correct':True,'input_unchanged':True});print(rows[-1],flush=True)
assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
torch.npu.synchronize()
Path(__file__).with_name('nan-to-num-extended.json').write_text(json.dumps({'rows':rows,'runtime_limit':torch.npu.get_device_limit(0),'module_sha256':frozen},indent=2)+'\n')
