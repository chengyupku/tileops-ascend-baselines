"""Dropout large tails, replay, changed data, and cross-tile RNG checks."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import dropout as module
from tileops.ops.elementwise import DropoutFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows=[]
def verify(y,x,p):
 compare_outputs(y,x/(1-p),ElementwiseWorkload('DropoutFwdOp',(x,),p=p).verification(x))
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 tile=4096 if dtype==torch.float32 else 8192
 for n in [32767,524289,540671,2*tile+256]:
  p=.3 if n==2*tile+256 else .5
  op=DropoutFwdOp(p=p,seed=42)
  x=torch.ones(n,dtype=dtype);a=x.npu();actual=op(a).cpu();verify(actual,x,p)
  assert torch.equal(a.cpu(),x),'input mutated'
  assert torch.equal(actual,op(a).cpu()),'replay changed'
  changed=2*x;b=changed.npu();second=op(b).cpu();verify(second,changed,p)
  assert torch.equal(actual!=0,second!=0),'mask depends on input values'
  assert torch.equal(b.cpu(),changed),'changed input mutated'
  mask=actual!=0
  assert not torch.equal(mask[:tile],mask[tile:2*tile]),'consecutive tile replay'
  if n>65*tile:
   assert not torch.equal(mask[:tile],mask[64*tile:65*tile]),'persistent iteration replay'
  if n==2*tile+256:
   tail=mask[-256:]
   for start in range(0,2*tile,256):
    assert not torch.equal(tail,mask[start:start+256]),'tail replayed full-tile subsequence'
   other=DropoutFwdOp(p=p,seed=99)(a).cpu();verify(other,x,p)
   assert not torch.equal(mask,other!=0),'different seeds identical'
  rows.append({'dtype':str(dtype),'n':n,'p':p,'passed':True,'replay':True,'changed_input':True,'cross_tile_checked':True,'tail_collision_checked':n==2*tile+256});print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('dropout-tail-stress.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
