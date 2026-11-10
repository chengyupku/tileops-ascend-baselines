import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.ops.elementwise import MulFwdOp
from tileops.kernels.ascend.elementwise import arithmetic_mul as module
x=torch.arange(256,dtype=torch.uint8).repeat_interleave(256);b=torch.arange(256,dtype=torch.uint8).repeat(256)
expected=(x!=0)&(b!=0)
a=x.npu().view(torch.bool);c=b.npu().view(torch.bool)
got=MulFwdOp()(a,c).cpu();cpu=torch.mul(x.view(torch.bool),b.view(torch.bool));logical=torch.logical_and(x.view(torch.bool),b.view(torch.bool))
bad=(got!=expected).nonzero().flatten()
row={'pairs':65536,'cpu_mul_mismatches':int((cpu!=expected).sum()),'cpu_logical_mismatches':int((logical!=expected).sum()),'op_mismatches':bad.numel(),'examples':[(int(x[i]),int(b[i]),bool(got[i]),bool(expected[i])) for i in bad[:8]],'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)}
assert torch.equal(a.view(torch.uint8).cpu(),x) and torch.equal(c.view(torch.uint8).cpu(),b)
row['input_unchanged']=True;torch.npu.synchronize();Path(__file__).with_name('mul-bool-probe.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
assert row['op_mismatches']==0,'Mul bool truthiness differs'
