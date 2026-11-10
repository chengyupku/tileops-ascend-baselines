"""CPU coordinator: real shape/dtype configuration comparisons, one guard each."""
import json
from pathlib import Path
import subprocess
import time
import argparse
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(HERE.parents[3])
p=argparse.ArgumentParser();p.add_argument('--tag',default='config15');p.add_argument('--op',action='append');p.add_argument('--case',action='append');a=p.parse_args()
rows=[]
for op in ('Log1pFwdOp','LogFwdOp','SqrtFwdOp','RsqrtFwdOp'):
    if a.op and op not in a.op:continue
    for case in ('16M-float16','16M-bfloat16','16M-float32','256M-float16','256M-bfloat16'):
        if a.case and case not in a.case:continue
        label=a.tag+'-'+op+'-'+case
        output=str(REL/'search'/label)
        command=['python3',str(HERE/'remote.py'),'--label',label,'--timeout','600','--',str(REL/'search_complete.py'),'--op',op,'--case',case,'--output',output]
        start=time.monotonic();rc=subprocess.run(command).returncode
        rows.append({'label':label,'rc':rc,'wall_seconds':time.monotonic()-start,'output':output})
        (HERE/(a.tag+'-batch-status.json')).write_text(json.dumps(rows,indent=2))
        if rc:raise SystemExit(rc)
