import datetime,json,os,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
idx=int(sys.argv[1]);x=json.loads((P/'SCOPE.json').read_text())[idx];d=P/(sys.argv[2] if len(sys.argv)>2 else f'case{idx}')
cmd=['npu-guard','run','--owner','baseline700-20261026/nn_remote_remaining6','--wait','3600','--timeout','300','--','/root/dyq/workspace/.tools/baselines/baseline-run',x['baseline'],'python',str(P/'run_case.py'),x['baseline'],x['operator'],x['case_id'],d.name]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
(d/'COMMAND.json').write_text(json.dumps(cmd));start=time.time();env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (d/'guard.log').open('w') as f:
 proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=now(),current_key=[x[k] for k in ('baseline','operator','case_id')],phase='waiting or running guarded checker',elapsed_s=time.time()-start)))
  try:proc.wait(timeout=30)
  except subprocess.TimeoutExpired:pass
rc=proc.returncode
(d/'GUARD_RESULT.json').write_text(json.dumps(dict(returncode=rc,elapsed_s=time.time()-start,finished_at=now()),indent=2));print('guard',rc)
if (d/'checker.json').exists():
 a=json.loads((d/'checker.json').read_text());print(json.dumps({k:a.get(k) for k in ('status','stage','error','api','verification')})[:3000])
sys.exit(rc)
