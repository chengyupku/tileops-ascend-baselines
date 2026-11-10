"""Probe ReLU IEEE semantics against CPU reference without changing public contracts."""
from pathlib import Path
import hashlib,json
import torch,torch_npu
from tileops.kernels.ascend.elementwise import relu as module
from tileops.ops.elementwise import ReluFwdOp
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 info=torch.finfo(dtype)
 values=torch.tensor([0.,-0.,1.,-1.,float('nan'),float('inf'),-float('inf'),info.tiny,-info.tiny,info.tiny*info.eps,-info.tiny*info.eps,info.max,-info.max],dtype=dtype)
 for n in [64,65536]:
  x=values.repeat((n+len(values)-1)//len(values))[:n].clone();a=x.npu();out=ReluFwdOp()(a).cpu();ref=torch.relu(x)
  equal=(out==ref)|(torch.isnan(out)&torch.isnan(ref));zero=(ref==0)&(out==0)
  storage=torch.int16 if dtype.itemsize==2 else torch.int32
  row=dict(dtype=str(dtype),n=n,mismatches=int((~equal).sum()),zero_sign_mismatches=int((torch.signbit(out[zero])!=torch.signbit(ref[zero])).sum()),input_unchanged=torch.equal(a.cpu().view(storage),x.view(storage)))
  rows.append(row);print(row,flush=True)
torch.npu.synchronize()
Path(__file__).with_name('relu-special-probe.json').write_text(json.dumps(dict(checks=rows,runtime_limit=torch.npu.get_device_limit(0),module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()),indent=2)+'\n')
assert all(v['mismatches']==0 and v['zero_sign_mismatches']==0 and v['input_unchanged'] for v in rows)
