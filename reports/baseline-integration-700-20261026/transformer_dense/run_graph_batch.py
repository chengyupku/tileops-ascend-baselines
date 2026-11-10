import subprocess,sys
from pathlib import Path
p=Path(__file__).parent
for name in ['run_longrope.py','run_topk.py','run_gemm_compile22.py']:
 r=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)])
 print(name,r.returncode,flush=True)
