import subprocess,time,json
from pathlib import Path
p=Path(__file__).resolve().parent;t=time.monotonic();results=[]
for name in ['run_topk.py','run_rope.py','run_norm.py']:
    if name=='run_norm.py' and time.monotonic()-t>150:break
    rc=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]).returncode
    results.append({'script':name,'returncode':rc,'elapsed_s':time.monotonic()-t})
    (p/'small_families.json').write_text(json.dumps(results,indent=2))
