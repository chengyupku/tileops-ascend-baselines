
import json,hashlib
from pathlib import Path
root=Path('/opt/tileops-all-pairs/workspace')
old=root/'reports/baseline-integration-700-20261026/threehost_bmm_input_backend1'
torch=root/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages/torch'
files=[(old/'TileOPs/workloads/gemm.py',[(364,389)]),(old/'TileOPs/scripts/ascend_pair_check.py',[(137,162),(320,345)]),(old/'run.sh',[(1,100)]),(old/'FROZEN.json',[(1,250)]),(torch/'backends/mkldnn/__init__.py',[(1,160)]),(torch/'include/ATen/Context.h',[(1,9999)]),(torch/'_torch_docs.py',[(1,99999)])]
out={}
for p,ranges in files:
 raw=p.read_bytes();lines=raw.decode().splitlines(); excerpts=[]
 if p.name=='Context.h':
  ranges=[(max(1,i-3),min(len(lines),i+5)) for i,l in enumerate(lines,1) if any(t in l for t in ['enabled_mkldnn','allowBF16Reduction','float32_matmul_precision','Float32Precision::NONE'])]
 if p.name=='_torch_docs.py':
  hits=[i for i,l in enumerate(lines,1) if l.strip()=='torch.bmm,'];ranges=[(i,i+85) for i in hits]
 for lo,hi in ranges:excerpts.append(dict(first=lo,text='\n'.join(lines[lo-1:hi])))
 out[str(p)]=dict(sha256=hashlib.sha256(raw).hexdigest(),excerpts=excerpts)
print(json.dumps(out,indent=2))
