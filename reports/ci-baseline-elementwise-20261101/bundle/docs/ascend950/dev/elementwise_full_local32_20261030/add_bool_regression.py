import torch,torch_npu,json
from pathlib import Path
from tileops.ops.elementwise import AddFwdOp
from tileops.kernels.ascend.elementwise import arithmetic as module
import hashlib
rows=[]
x=torch.arange(256,dtype=torch.uint8).repeat_interleave(256);b=torch.arange(256,dtype=torch.uint8).repeat(256)
a=x.npu().view(torch.bool);c=b.npu().view(torch.bool)
for alpha in [0,1,2]:
 expected=(x!=0)|((b!=0) if alpha else torch.zeros_like(b,dtype=torch.bool))
 cpu=torch.add(x.view(torch.bool),b.view(torch.bool),alpha=alpha)
 got=AddFwdOp(alpha=alpha)(a,c).cpu();bad=(got!=expected).nonzero().flatten()
 row={'alpha':alpha,'pairs':65536,'cpu_mismatches':int((cpu!=expected).sum()),'npu_mismatches':bad.numel(),'examples':[(int(x[i]),int(b[i]),bool(got[i]),bool(expected[i])) for i in bad[:8]]};assert row['cpu_mismatches']==row['npu_mismatches']==0;rows.append(row);print(row,flush=True)
torch.npu.synchronize();Path(__file__).with_name('add-bool-regression.json').write_text(json.dumps({'checks':rows,'module_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
