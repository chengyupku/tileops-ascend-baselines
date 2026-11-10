import torch,torch_npu,json
from pathlib import Path
rows=[]
x=torch.arange(256,dtype=torch.uint8).repeat_interleave(256);b=torch.arange(256,dtype=torch.uint8).repeat(256)
a=x.npu().view(torch.bool);c=b.npu().view(torch.bool)
for alpha in [0,1,2]:
 expected=(x!=0)|((b!=0) if alpha else torch.zeros_like(b,dtype=torch.bool))
 cpu=torch.add(x.view(torch.bool),b.view(torch.bool),alpha=alpha)
 got=torch.add(a,c,alpha=alpha).cpu();bad=(got!=expected).nonzero().flatten()
 row={'alpha':alpha,'cpu_mismatches':int((cpu!=expected).sum()),'npu_mismatches':bad.numel(),'examples':[(int(x[i]),int(b[i]),bool(got[i]),bool(expected[i])) for i in bad[:8]]};rows.append(row);print(row,flush=True)
torch.npu.synchronize();Path(__file__).with_name('add-bool-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
