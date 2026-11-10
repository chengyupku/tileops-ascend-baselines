"""Exercise direct broadcast paths on short rows, padding and raw bool bytes."""
import json
from pathlib import Path
import torch
import torch_npu
from tileops.ops.elementwise import BitwiseAndFwdOp
rows=[]
for dtype in [torch.bool,torch.int32,torch.int64]:
 for channels,repeat in [(4,4),(4,64),(4,65),(8,128),(3,64),(4,3136)]:
  shape=(2,channels,repeat)
  if dtype==torch.bool:
   raw=torch.randint(0,256,shape,dtype=torch.uint8)
   other=torch.randint(0,256,(channels,1),dtype=torch.uint8)
   raw.reshape(-1)[::3]=0;other.reshape(-1)[::2]=0
   x=raw.npu().view(torch.bool);y=other.npu().view(torch.bool)
   ref=torch.logical_and(raw!=0,other!=0)
  else:
   raw=torch.randint(-100000,100000,shape,dtype=dtype)
   other=torch.randint(-100000,100000,(channels,1),dtype=dtype)
   raw.reshape(-1)[0]=torch.iinfo(dtype).min;other.reshape(-1)[0]=-1
   x=raw.npu();y=other.npu();ref=torch.bitwise_and(raw,other)
  op=BitwiseAndFwdOp();out=op(x,y);torch.npu.synchronize()
  torch.testing.assert_close(out.cpu(),ref,rtol=0,atol=0)
  if dtype==torch.bool:torch.testing.assert_close(out.view(torch.uint8).cpu(),ref.to(torch.uint8),rtol=0,atol=0)
  kernels=[k for es in op._built_entries.values() for k in es.values()]
  rows.append({'dtype':str(dtype),'shape':shape,'correct':True,'fast':kernels[0].fast})
Path(__file__).with_name('and-broadcast-extra.json').write_text(json.dumps(rows,indent=2)+'\n')
print(len(rows),'checks passed',flush=True)
