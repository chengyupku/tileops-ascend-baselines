import subprocess,time,json
from pathlib import Path
p=Path(__file__).resolve().parent;t=time.monotonic();results=[]
for name in ['run_rope.py','run_gmm_check.py']:
    if name=='run_gmm_check.py' and time.monotonic()-t>150:break
    rc=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]).returncode
    results.append({'script':name,'returncode':rc,'elapsed_s':time.monotonic()-t})
    (p/'repaired_batch.json').write_text(json.dumps(results,indent=2))
