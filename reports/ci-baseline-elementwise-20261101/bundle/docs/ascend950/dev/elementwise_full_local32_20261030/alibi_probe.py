import json
from pathlib import Path
import torch,torch_npu
from tileops.kernels.ascend.elementwise.alibi import AscendAlibiKernel
from workloads.elementwise import alibi_reference
r=Path(__file__).resolve().parent
rows=[]
for s,h in [(8,2),(128,8)]:
 k=AscendAlibiKernel(s,h,torch.float32,'npu');y=k().cpu();ref=alibi_reference(s,h,torch.float32,'cpu')
 (r/f'alibi-probe-{s}-{h}.c').write_text(k.compiled.get_kernel_source())
 row=dict(s=s,h=h,got=y[:2,:2,:8].tolist(),ref=ref[:2,:2,:8].tolist(),nan=int(y.isnan().sum()),inf=int(y.isinf().sum()))
 rows.append(row);print(row,flush=True)
torch.npu.synchronize();(r/'alibi-probe.json').write_text(json.dumps(rows,indent=2)+'\n')
