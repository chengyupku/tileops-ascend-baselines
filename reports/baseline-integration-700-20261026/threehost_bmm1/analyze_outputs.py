"""CPU-only supplemental statistics over saved outputs; never changes canonical verdict."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,'/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
import numpy as np
P=Path(__file__).resolve().parent;D=P/'square-4k-bfloat16'
a=np.memmap(D/'candidate_0_output.bin',dtype='<u2',mode='r');b=np.memmap(D/'reference_0_output.bin',dtype='<u2',mode='r');assert a.size==b.size==67108864
rows=[]
for start in range(0,a.size,1048576):
 x=(a[start:start+1048576].astype(np.uint32)<<16).view(np.float32);y=(b[start:start+1048576].astype(np.uint32)<<16).view(np.float32)
 diff=np.abs(x-y);bad=~np.isclose(x,y,rtol=.016,atol=.016,equal_nan=True)
 rows.append(dict(start=start,count=x.size,mismatched=int(bad.sum()),max_abs=float(diff.max()),nonfinite_candidate=int((~np.isfinite(x)).sum()),nonfinite_reference=int((~np.isfinite(y)).sum())))
out=dict(kind='supplemental CPU float32 comparison of stored BF16 outputs, not a replacement checker',elements=a.size,mismatched=sum(r['mismatched'] for r in rows),max_abs=max(r['max_abs'] for r in rows),blocks=rows,sha256={name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ['candidate_0_output.bin','reference_0_output.bin']})
(P/'FULL_OUTPUT_STATS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['blocks','sha256']}))
