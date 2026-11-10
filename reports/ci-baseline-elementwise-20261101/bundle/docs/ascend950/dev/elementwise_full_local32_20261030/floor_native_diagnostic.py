import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import floor
from tileops.ops.elementwise import FloorFwdOp
rows=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 x=torch.tensor([0.,-0.,1e-45,-1e-45,torch.finfo(dtype).tiny/2,-torch.finfo(dtype).tiny/2,.5,-.5,float('inf'),-float('inf'),float('nan')],dtype=dtype)
 y=FloorFwdOp()(x.npu()).cpu();ref=torch.floor(x)
 rows.append({'dtype':str(dtype),'samples':[{'input':str(a.item()),'actual':str(b.item()),'reference':str(c.item())} for a,b,c in zip(x,y,ref)]})
 print(rows[-1],flush=True)
torch.npu.synchronize()
Path(__file__).with_name('floor-native-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
