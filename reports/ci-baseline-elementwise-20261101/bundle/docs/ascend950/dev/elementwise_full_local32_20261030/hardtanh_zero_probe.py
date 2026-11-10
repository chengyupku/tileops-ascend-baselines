import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import hardtanh
from tileops.ops.elementwise import HardtanhFwdOp
rows=[]
for lo,hi in [(0.,1.),(-0.,1.),(-1.,0.),(-1.,-0.),(-0.,0.)]:
 x=torch.tensor([0.,-0.,-1.,1.,-float('inf'),float('inf'),float('nan'),-1e-7],dtype=torch.float16).repeat(9)[:65]
 op=HardtanhFwdOp(min_val=lo,max_val=hi);out=op(x.npu()).cpu();ref=torch.nn.functional.hardtanh(x,min_val=lo,max_val=hi)
 row={'lo':lo.hex(),'hi':hi.hex(),'x':x[:8].tolist(),'out_bits':out[:8].view(torch.int16).tolist(),'ref_bits':ref[:8].view(torch.int16).tolist()};rows.append(row);print(json.dumps(row),flush=True)
torch.npu.synchronize()
Path(__file__).with_name('hardtanh-zero-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
