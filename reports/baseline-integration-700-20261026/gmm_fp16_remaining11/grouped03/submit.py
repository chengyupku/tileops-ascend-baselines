import subprocess,time,json
from pathlib import Path
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/grouped03'
cmd=f'npu-guard run --owner baseline700-20261026/gmm_fp16_remaining11 --wait 3600 --timeout 300 -- bash {r}/run.sh > {r}/guard.log 2>&1; rc=$?; echo "$rc" > {r}/guard.exit; exit "$rc"'
(P/'SUBMISSION.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'argv':[ssh,'openlibing-DevEnv-661495',cmd],'api_queue_s':None},indent=2)+'\n')
with (P/'transport.log').open('w') as log:
 p=subprocess.Popen([ssh,'openlibing-DevEnv-661495',cmd],stdout=log,stderr=subprocess.STDOUT)
 while True:
  (P/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'phase':'guard_pending_or_running','current_key':['ops-transformer','GroupedGemmFwdOp','moe-wgrad-float16'],'queue_s':None,'pid':p.pid})+'\n')
  try:rc=p.wait(timeout=30);break
  except subprocess.TimeoutExpired:pass
(P/'transport.exit').write_text(str(rc)+'\n')
print('transport exit',rc,flush=True)
