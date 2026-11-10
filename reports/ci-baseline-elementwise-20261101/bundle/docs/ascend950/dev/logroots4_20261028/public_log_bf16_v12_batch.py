import json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;rel=p.relative_to(p.parents[3]);rows=[]
for case in ['16M-bfloat16','256M-bfloat16']:
 label='public-log-bf16-v12-'+case
 rc=subprocess.run(['python3',str(p/'remote.py'),'--label',label,'--timeout','600','--',str(rel/'search_public_log_bf16_v12.py'),'--op','LogFwdOp','--case',case]).returncode
 rows.append({'case_id':case,'label':label,'rc':rc})
 (p/'public-log-bf16-v12-status.json').write_text(json.dumps(rows,indent=2))
 if rc:raise SystemExit(rc)
subprocess.run(['python3',str(p/'fetch.py'),'search','edges','baseline','environment'],check=True)
