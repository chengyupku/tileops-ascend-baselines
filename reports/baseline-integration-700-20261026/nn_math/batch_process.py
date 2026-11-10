import json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;jobs=json.loads(Path(sys.argv[1]).read_text());started=time.monotonic()
for base,op,cid,attempt in jobs:
 if time.monotonic()-started>245:break
 out=P/attempt;t=time.time()
 with (out/'case.log').open('w') as f:r=subprocess.run([sys.executable,str(P/'run_case.py'),base,op,cid,attempt],stdout=f,stderr=subprocess.STDOUT)
 (out/'CHILD_RESULT.json').write_text(json.dumps(dict(returncode=r.returncode,elapsed_s=time.time()-t)))
 print(attempt,r.returncode,flush=True)
 if r.returncode and len({tuple(j[:2]) for j in jobs})==1:break
