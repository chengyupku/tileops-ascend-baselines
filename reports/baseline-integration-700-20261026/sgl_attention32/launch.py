from pathlib import Path
import sys,json,os,subprocess,datetime,time
R=Path(__file__).resolve().parent
op,cid,attempt=sys.argv[1:]
a=R/'runs'/attempt;a.mkdir(parents=True,exist_ok=False)
import shutil
for source in R.glob('*.py'): shutil.copy2(source,a/source.name)
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
t=time.monotonic(); started=now()
(R/'HEARTBEAT.json').write_text(json.dumps(dict(utc=started,phase='guard_wait_or_run',current_keys=[cid],last_progress_at=started,blocked_reason=None)))
cmd=['npu-guard','run','--owner','baseline700-20261026/sgl_attention32','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run','sgl-kernel-npu','python',str(R/'run_case.py'),'sgl-kernel-npu',op,cid,str(a/'output')]
(a/'COMMAND.json').write_text(json.dumps(cmd)); env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
with (a/'guard.log').open('x') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(a/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=p.returncode,started=started,finished=now(),elapsed_seconds=time.monotonic()-t)))
print(a,p.returncode)
