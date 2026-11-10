from pathlib import Path
import subprocess,json,time,datetime
P=Path(__file__).resolve().parent
keys=[r['case_id'] for r in json.loads((P/'SCOPE.json').read_text()) if r['baseline']=='flash-linear-attention-npu' and r['operator']=='GDNFwdOp' and r['status']=='blocked' and r['case_id']!='decode-b8-float16']
for cid in keys:
 old=P/('fla_'+cid)/'GUARD_RESULT.json'
 if old.exists() and json.loads(old.read_text())['returncode']!=75:continue
 label=['readmission1'] if old.exists() else []
 p=subprocess.Popen(['python3',str(P/'local_fla_run.py'),cid]+label);last=0
 while p.poll() is None:
  if time.monotonic()-last>30:
   (P/'HEARTBEAT.local.json').write_text(json.dumps(dict(phase='exact_fla_fullchecker',currentkey=cid,host='local',lastprogress=datetime.datetime.now(datetime.timezone.utc).isoformat(),blockedreason=None)));last=time.monotonic()
  time.sleep(1)
 d=json.loads((P/('fla_'+cid+('_readmission1' if label else ''))/'GUARD_RESULT.json').read_text());print(cid,d['returncode'],flush=True)
 if d['returncode'] in [75,76,78]:break
