"""CPU coordinator: one bounded foreground guard request at a time."""
import argparse
import json
from pathlib import Path
import subprocess
import time
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(HERE.parents[3])
p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['primary','tla','catlass','compile'],required=True)
p.add_argument('--op',action='append')
p.add_argument('--case',action='append',help='Select individual compile or CATLASS cases')
p.add_argument('--force',action='store_true',help='Explicitly repeat an already completed guarded attempt')
a=p.parse_args()
if a.case and a.phase not in {'catlass','compile'}:p.error('--case is supported for compile and catlass phases')
ops=['LogFwdOp','Log1pFwdOp','SqrtFwdOp','RsqrtFwdOp']
jobs=[]
if a.phase=='primary':
    for op in ops:
        for library,env,script in [('ops-math','env-math.sh','baselines.py'),('torch','env.sh','baseline_torch.py'),('ops-nn','env-nn.sh','baselines.py')]:
            jobs.append((op,library,env,script,'',600))
elif a.phase=='tla':
    for op in ['LogFwdOp','SqrtFwdOp','RsqrtFwdOp']:
        jobs.append((op,'tilelang-ascend','env-tla.sh','baselines.py','',600))
elif a.phase=='catlass':
    # Many original 1024-element launches: reserve each manifest case separately.
    for op in ['LogFwdOp','SqrtFwdOp']:
        for case in ['16M-float16','16M-bfloat16','16M-float32','256M-float16','256M-bfloat16']:
            jobs.append((op,'catlass','env-catlass.sh','baseline_catlass.py','^'+case+'$',900))
else:
    for op in ops:
        for case in ['16M-float16','16M-bfloat16','16M-float32','256M-float16','256M-bfloat16']:
            jobs.append((op,'torch-compile','env-compile.sh','baseline_torch.py','^'+case+'$',900))
records=[]
for index,(op,library,env,script,regex,timeout) in enumerate(jobs):
    if a.op and op not in a.op:continue
    if a.case and regex.removeprefix('^').removesuffix('$') not in a.case:continue
    label='baseline-'+a.phase+'-'+op+'-'+library+'-'+str(index)
    output=str(REL/'baseline'/(label+'.jsonl'))
    # A completed attempt remains reviewable even if its original implementation
    # rejected a case. Qualification is decided by audit.py, not this scheduler.
    # Explicit --force is required to repeat such an attempt.
    receipts=[json.loads(path.read_text()) for path in (HERE/'requests').glob(label+'_*/status.json')]
    existing=HERE/'baseline'/(label+'.jsonl')
    if receipts and existing.exists() and not a.force:
        latest=max(receipts,key=lambda row:row['completed_at'])
        prior=[json.loads(line) for line in existing.read_text().splitlines()]
        expected_count=1 if regex else 5
        if latest['guard_exit_code']==0 and len({row['case_id'] for row in prior})==expected_count:
            records.append({'label':label,'rc':0,'output':output,'wall_seconds':0,
                            'reused_completed_attempt':latest['started_utc']})
            (HERE/('baseline-'+a.phase+'-status.json')).write_text(json.dumps(records,indent=2))
            print('REUSE_COMPLETED_ATTEMPT',label,flush=True)
            continue
    command=['python3',str(HERE/'remote.py'),'--label',label,'--library',library,'--env',env,'--timeout',str(timeout),'--',str(REL/script),'--op',op,'--library',library,'--output',output]
    if regex:command+=['--regex',regex]
    if library == 'catlass':command+=['--exported-flush']
    begin=time.monotonic();rc=subprocess.run(command).returncode
    records.append({'label':label,'rc':rc,'output':output,'wall_seconds':time.monotonic()-begin})
    (HERE/('baseline-'+a.phase+'-status.json')).write_text(json.dumps(records,indent=2))
    if rc:raise SystemExit(rc)
