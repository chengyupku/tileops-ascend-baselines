"""Public PReLU empty channels and zero-sized spatial dimensions."""
import json,hashlib
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from tileops.kernels.ascend.elementwise import prelu as module
from tileops.ops.elementwise import PreluFwdOp
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for shape,c in [((2,0,3),0),((0,0,3),0),((0,3,0),3)]:
  x=torch.empty(shape,dtype=dtype);w=torch.ones(c,dtype=dtype)
  expected=F.prelu(x,w);a=x.npu();b=w.npu();result=PreluFwdOp()(a,b)
  assert result.shape==expected.shape and result.dtype==dtype and result.device==a.device
  assert result.numel()==0 and result is not a
  rows.append({'dtype':str(dtype),'shape':shape,'channels':c,'passed':True});print(rows[-1],flush=True)
 try:module.AscendPreluKernel(1,dtype,0,1)
 except ValueError:pass
 else:raise AssertionError('nonempty input with zero weights accepted')
torch.npu.synchronize()
Path(__file__).with_name('prelu-empty-channels.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
