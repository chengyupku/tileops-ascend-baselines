import torch,torch_npu
from pathlib import Path
from tileops.kernels.ascend.elementwise import softplus
from tileops.ops.elementwise import SoftplusFwdOp
x=torch.tensor([0.,-0.,1.,-1.,float('inf'),-float('inf'),float('nan')],dtype=torch.float16).repeat(10)[:65]
for b in [0.,-0.]:
 op=SoftplusFwdOp(beta=b);y=op(x.npu()).cpu();ref=torch.nn.functional.softplus(x,beta=b)
 print(b,'got',y[:7].tolist(),'ref',ref[:7].tolist(),flush=True)
 k=[k for e in op._built_entries.values() for k in e.values()][0]
 Path(__file__).with_name('softplus-zero-'+b.hex()+'.c').write_text(k.kernel.get_kernel_source())
torch.npu.synchronize()
