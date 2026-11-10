"""Serial guarded requests for frozen source acceptance; no device work here."""
import argparse
import json
from pathlib import Path
import subprocess
import time
HERE=Path(__file__).resolve().parent
REL=HERE.relative_to(HERE.parents[3])
p=argparse.ArgumentParser();p.add_argument('--version',default='v1');p.add_argument('--op',action='append')
a=p.parse_args();records=[]
ops={'Log1pFwdOp':'log1p','LogFwdOp':'log','SqrtFwdOp':'sqrt','RsqrtFwdOp':'rsqrt'}
def run(label,script,args,*,env='env.sh',timeout=360,allow_failure=False):
    cmd=['python3',str(HERE/'remote.py'),'--label',label,'--env',env,'--timeout',str(timeout),'--',str(REL/script),*args]
    start=time.monotonic();rc=subprocess.run(cmd).returncode
    records.append({'label':label,'rc':rc,'wall_seconds':time.monotonic()-start,'argv':cmd})
    (HERE/('formal-batch-'+a.version+'-status.json')).write_text(json.dumps(records,indent=2))
    if rc and not allow_failure:raise SystemExit(rc)
for op,kind in ops.items():
    if a.op and op not in a.op:continue
    tag=op+'-'+a.version
    run('final-'+tag,'pilot.py',['--op',op,'--output',str(REL/'final'/('final-'+tag+'.jsonl'))],timeout=600)
    for library,env,script in [('ops-math','env-math.sh','baselines.py'),('torch','env.sh','baseline_torch.py'),('ops-nn','env-nn.sh','baselines.py')]:
        run('baseline-'+library+'-'+tag,script,['--op',op,'--library',library,'--output',str(REL/'baseline'/('baseline-'+library+'-'+tag+'.jsonl'))],env=env,timeout=600)
    run('edges-'+tag,'edges.py',['--op',op,'--output',str(REL/'edges'/('edges-'+tag+'.json'))],timeout=600)
    run('pytest-'+tag,'invoke_pytest.py',['--kernel-module',kind,'TileOPs/tests/ops/test_unary_math.py::test_'+kind,'TileOPs/tests/ops/test_unary_math.py::test_'+kind+'_edge','-q','--junitxml='+str(REL/'edges'/('pytest-'+tag+'.xml'))],timeout=360,allow_failure=True)
    run('profile-'+tag,'diagnose.py',['--op',op,'--output',str(REL/'profiles'/tag)],timeout=240)
