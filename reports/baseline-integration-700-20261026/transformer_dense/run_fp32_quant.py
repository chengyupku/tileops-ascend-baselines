import subprocess
from pathlib import Path
p=Path(__file__).parent
for name in ['run_gmm_fp32.py','run_quant.py']:
 r=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]);print(name,r.returncode,flush=True)
