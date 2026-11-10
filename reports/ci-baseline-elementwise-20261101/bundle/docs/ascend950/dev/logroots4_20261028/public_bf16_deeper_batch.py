import json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;rel=p.relative_to(p.parents[3]);rows=[]
jobs=[('public-configs-LogFwdOp-bf16-deeper','search_public_bf16_deeper.py',[]),('public-log-bf16-v10-deeper-16M-bfloat16','search_public_log_bf16_v10_deeper.py',['--case','16M-bfloat16'])]
for label,script,extra in jobs:
 rc=subprocess.run(['python3',str(p/'remote.py'),'--label',label,'--timeout','600','--',str(rel/script),'--op','LogFwdOp',*extra]).returncode
 rows.append({'label':label,'rc':rc});(p/'public-bf16-deeper-status.json').write_text(json.dumps(rows,indent=2))
 if rc:raise SystemExit(rc)
subprocess.run(['python3',str(p/'fetch.py'),'search','edges'],check=True)
