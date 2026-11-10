import json,hashlib
from pathlib import Path
import torch,torch_npu
from tileops.ops.elementwise import GeFwdOp
from tileops.kernels.ascend.elementwise import comparison as module
x=torch.arange(256,dtype=torch.uint8).repeat_interleave(256);b=torch.arange(256,dtype=torch.uint8).repeat(256)
expected=(x!=0)>=(b!=0)
a=x.npu().view(torch.bool);c=b.npu().view(torch.bool)
got=GeFwdOp()(a,c).cpu();cpu=torch.ge(x.view(torch.bool),b.view(torch.bool));logical=torch.ge(x.view(torch.bool),b.view(torch.bool))
bad=(got!=expected).nonzero().flatten()
row={'pairs':65536,'cpu_ne_mismatches':int((cpu!=expected).sum()),'cpu_ne_duplicate_mismatches':int((logical!=expected).sum()),'op_mismatches':bad.numel(),'examples':[(int(x[i]),int(b[i]),bool(got[i]),bool(expected[i])) for i in bad[:8]],'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0),'dependency_manifest':{str(Path(module.__file__).with_name(name)):hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest() for name in []}}
assert torch.equal(a.view(torch.uint8).cpu(),x) and torch.equal(c.view(torch.uint8).cpu(),b)
assert bool((got.view(torch.uint8)<=1).all())
row['input_unchanged']=True;torch.npu.synchronize();Path(__file__).with_name('ge-bool-probe.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
assert row['op_mismatches']==0,'Ge bool truthiness differs'
