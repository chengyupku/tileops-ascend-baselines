from pathlib import Path
import subprocess,json,time,sys
P=Path(__file__).resolve().parent;op,cid=sys.argv[1:3];suffix=sys.argv[3] if len(sys.argv)>3 else '_local';key=['tilelang-ascend',op,cid];tag='_'.join(key)+suffix
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_pto_original23','--wait','120','--timeout','300','--','bash',str(P/'local_run.sh'),op,cid,suffix]
r=dict(key=key,host='local',argv=cmd,started=time.time());log=P/(tag+'.guard.log');assert not log.exists()
with log.open('w') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 while True:
  (P/'LOCAL_HEARTBEAT.json').write_text(json.dumps(dict(host='local',phase='guard_running' if 'acquired device' in log.read_text() else 'guard_wait',exactkey=key,updated_at=time.time(),log_tail=log.read_text()[-1200:]),indent=2))
  try:rc=proc.wait(timeout=20);break
  except subprocess.TimeoutExpired:pass
r.update(returncode=rc,finished=time.time());(P/(tag+'.guard.json')).write_text(json.dumps(r,indent=2));print(json.dumps(r))
