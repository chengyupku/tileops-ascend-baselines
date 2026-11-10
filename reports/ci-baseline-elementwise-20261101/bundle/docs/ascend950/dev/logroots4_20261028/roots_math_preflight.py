import json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;rel=p.relative_to(p.parents[3]);rows=[]
for op in ['SqrtFwdOp','RsqrtFwdOp']:
 label='baseline-math-'+op+'-bf16-preflight'
 rc=subprocess.run(['python3',str(p/'remote.py'),'--label',label,'--library','ops-math','--env','env-math.sh','--timeout','180','--',str(rel/'baselines.py'),'--op',op,'--library','ops-math','--regex','^16M-bfloat16$','--output',str(rel/'baseline'/(label+'.jsonl'))]).returncode
 rows.append({'op':op,'label':label,'rc':rc})
 (p/'roots-math-preflight-status.json').write_text(json.dumps(rows,indent=2))
 if rc:raise SystemExit(rc)
subprocess.run(['python3',str(p/'fetch.py'),'baseline','environment'],check=True)
