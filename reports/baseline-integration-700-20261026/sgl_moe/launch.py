import sys,os,json,subprocess,time,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
op,cid,label=sys.argv[1:]; a=P/label;a.mkdir(exist_ok=False)
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_moe','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run','sgl-kernel-npu','python',str(P/'run_case.py'),op,cid,str(a)]
# workspace is three parents above task directory

(a/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
t=time.time();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (a/'guard.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
(a/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=r.returncode,started=t,ended=time.time(),elapsed=time.time()-t)))
raise SystemExit(r.returncode)
