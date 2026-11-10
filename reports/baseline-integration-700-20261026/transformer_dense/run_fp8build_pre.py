import subprocess
from pathlib import Path
p=Path(__file__).parent
for name in ['run_gemm_compilefp8.py','run_pre.py']:
 r=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]);print(name,r.returncode,flush=True)
