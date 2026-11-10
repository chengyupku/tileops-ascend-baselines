"""Record runtime budget and verify a small real kernel under the guard."""
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise.bitwise_binary import AscendBitwiseAndKernel
from tileops.ops.elementwise import BitwiseAndFwdOp
p=torch.npu.get_device_properties(0)
limit=torch.npu.get_device_limit(0)
assert limit == {'cube_core_num':32,'vector_core_num':64}, limit
x=torch.arange(4096,dtype=torch.int32).npu()
y=(torch.arange(4096,dtype=torch.int32)*7).npu()
out=BitwiseAndFwdOp()(x,y)
torch.npu.synchronize()
torch.testing.assert_close(out.cpu(),torch.bitwise_and(x.cpu(),y.cpu()),rtol=0,atol=0)
assert torch.npu.get_device_limit(0)==limit
result={'device':str(p),'runtime_limit':limit,'correct':True,'scope':'runtime resource budget; not physical 32-core SoC equivalence','host':__import__('socket').gethostname()}
Path(__file__).with_name('device-probe.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
