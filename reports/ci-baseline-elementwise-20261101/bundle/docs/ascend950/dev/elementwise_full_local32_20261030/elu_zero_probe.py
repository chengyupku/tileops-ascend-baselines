import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise import elu
from tileops.ops.elementwise import EluFwdOp
rows=[]
for alpha in [0.0,-0.0,-0.5,float('inf')]:
 x=torch.tensor([0.,-0.,-1.,1.,-float('inf'),float('inf'),float('nan'),-1e-7],dtype=torch.float16).repeat(9)[:65]
 op=EluFwdOp(alpha=alpha);out=op(x.npu()).cpu();ref=torch.nn.functional.elu(x,alpha=alpha)
 k=[v for entries in op._built_entries.values() for v in entries.values()][0]
 row={'alpha':alpha.hex(),'kernel_alpha':k.alpha.hex(),'x':x[:8].tolist(),'out':out[:8].tolist(),'ref':ref[:8].tolist(),'out_bits':out[:8].view(torch.int16).tolist(),'ref_bits':ref[:8].view(torch.int16).tolist()};rows.append(row);print(json.dumps(row),flush=True)
torch.npu.synchronize()
Path(__file__).with_name('elu-zero-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
