import json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
rel=p.relative_to(p.parents[3]);rows=[]
for op in ['SqrtFwdOp','RsqrtFwdOp']:
 label='public-configs-'+op+'-f32-r2'
 rc=subprocess.run(['python3',str(p/'remote.py'),'--label',label,'--timeout','360','--',str(rel/'search_public_roots_r2.py'),'--op',op]).returncode
 rows.append({'op':op,'label':label,'rc':rc})
 (p/'public-roots-r2-status.json').write_text(json.dumps(rows,indent=2))
 if rc:raise SystemExit(rc)
subprocess.run(['python3',str(p/'fetch.py'),'search'],check=True)
