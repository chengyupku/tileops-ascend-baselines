"""Arbitrary exponential shapes and strides under the original numerical contract."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'docs/ascend950/dev/lerp_exp4_20261028'))
import torch,torch_npu
from tileops.kernels.ascend.elementwise import exp,exponential
from tileops.ops.elementwise import ExpFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from full_digest import tensor_digest
rows=[];op=ExpFwdOp()
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for shape in [(0,3),(),(1,),(7,),(63,),(64,),(65,),(129,),(3,5),(2,1,5),(16387,)]:
  x=(torch.randn(shape)*3).to(dtype)
  if len(shape)==2 and x.numel():x=x.t().contiguous().t()
  ref=torch.exp(x);a=x.npu();before=tensor_digest([a]);out=op(a);torch.npu.synchronize()
  assert out.shape==x.shape and out.dtype==x.dtype
  if x.numel():
   compare_outputs(out.cpu(),ref,ElementwiseWorkload('ExpFwdOp',(a,)).verification(a))
   assert out.untyped_storage().data_ptr()!=a.untyped_storage().data_ptr()
  assert tensor_digest([a])==before
  rows.append({'dtype':str(dtype),'shape':shape,'correct':True,'input_unchanged':True})
  print(rows[-1],flush=True)
Path(__file__).with_name('exp-boundaries.json').write_text(json.dumps({'rows':rows,'runtime_limit':torch.npu.get_device_limit(0),'sources':{str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [exp,exponential]}},indent=2)+'\n')
