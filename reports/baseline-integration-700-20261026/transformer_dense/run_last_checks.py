import subprocess,time,json
from pathlib import Path
p=Path(__file__).parent;t=time.monotonic();rows=[]
for name in ['run_fp8_bmm_half.py','run_quant_prefill_down.py']:
 if time.monotonic()-t>210:break
 start=time.monotonic();r=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]);rows.append(dict(script=name,returncode=r.returncode,wall_s=time.monotonic()-start));print(name,r.returncode,flush=True)
(p/'LAST_CHECKS_BATCH.json').write_text(json.dumps(rows,indent=2))
