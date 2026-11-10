from pathlib import Path
import subprocess,json,time,shlex,datetime
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_isnan_unpack1'
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_isnan_unpack1','--wait','120','--timeout','180','--','/opt/tileops-all-pairs/bin/allpairs-python',remote+'/run_isnan.py','tilelang-ascend','IsnanFwdOp','16M-float32','_expanded']
r=dict(host='server2',argv=cmd,started=time.time())
with (P/'guard.log').open('x') as f:
 proc=subprocess.Popen([ssh,'openlibing-DevEnv-055409',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 while True:
  (P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='guard_wait_or_full_checker',host='server2',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),lastprogress=time.time(),accepted_added=0)))
  try:r['returncode']=proc.wait(timeout=20);break
  except subprocess.TimeoutExpired:pass
r['finished']=time.time();(P/'guard.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
