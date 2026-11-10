"""Arbitrary log1p shapes and strides under the original numerical contract."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/ascend950/dev/lerp_exp4_20261028'))
import torch,torch_npu
from tileops.kernels.ascend.elementwise import log1p as log1p_module
from tileops.ops.elementwise import Log1pFwdOp
from tileops.kernels.ascend.elementwise import _logroots
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from full_digest import tensor_digest
rows=[];op=Log1pFwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for shape in [(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,)]:
  x=(torch.randn(shape)*3).to(dtype)
  if len(shape)==2 and x.numel():x=x.t().contiguous().t()
  ref=torch.log1p(x);a=x.npu();before=tensor_digest([a]);out=op(a);torch.npu.synchronize()
  assert out.shape==x.shape and out.dtype==x.dtype
  if x.numel():
   compare_outputs(out.cpu(),ref,ElementwiseWorkload('Log1pFwdOp',(a,)).verification(a))
   assert out.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
  assert tensor_digest([a])==before
  rows.append({'dtype':str(dtype),'shape':shape,'correct':True,'input_unchanged':True})
  print(rows[-1],flush=True)
# Explicit output buffers check that unaligned tail stores stay within bounds.
import tilelang
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 n=16387;k=log1p_module.AscendLog1pKernel(n,dtype)
 explicit=tilelang.compile(k.compiled.prim_func,target='ascend',out_idx=[])
 x=torch.randn(n).to(dtype);a=x.npu()
 guarded=torch.full((n+128,),321,device='npu',dtype=dtype)
 output=guarded[64:64+n];output.fill_(0)
 explicit(a,output);torch.npu.synchronize()
 assert bool((guarded[:64]==321).all()) and bool((guarded[64+n:]==321).all())
 actual=output.view(dtype).cpu();ref=torch.log1p(x)
 compare_outputs(actual,ref,ElementwiseWorkload('Log1pFwdOp',(a,)).verification(a))
 bad=actual.clone();bad[-1]=float('nan')
 try:compare_outputs(bad,ref,ElementwiseWorkload('Log1pFwdOp',(a,)).verification(a))
 except AssertionError:pass
 else:raise AssertionError('checker accepted corrupt tail')
 rows.append({'dtype':str(dtype),'check':'sentinel_and_negative_control','correct':True})
Path(__file__).with_name('log1p-boundaries.json').write_text(json.dumps({'rows':rows,'runtime_limit':torch.npu.get_device_limit(0),'sources':{str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [log1p_module,_logroots]}},indent=2)+'\n')
