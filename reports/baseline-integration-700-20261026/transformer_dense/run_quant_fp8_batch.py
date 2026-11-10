import subprocess,time,json
from pathlib import Path
p=Path(__file__).parent;t=time.monotonic();records=[]
for name in ['run_quant_siblings.py','run_fp8_bmm.py','run_fp8_moe.py']:
 if time.monotonic()-t>220:
  records.append(dict(script=name,skipped='next short guard needed'));continue
 start=time.monotonic();r=subprocess.run(['/opt/tileops-all-pairs/bin/allpairs-python',str(p/name)]);records.append(dict(script=name,returncode=r.returncode,wall_s=time.monotonic()-start));print(name,r.returncode,flush=True)
(p/'QUANT_FP8_BATCH.json').write_text(json.dumps(records,indent=2))
