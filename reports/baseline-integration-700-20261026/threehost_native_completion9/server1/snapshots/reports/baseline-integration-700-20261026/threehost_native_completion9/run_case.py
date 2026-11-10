import sys,json,time,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;mode,op,cid=sys.argv[1:];D=P/cid;D.mkdir(exist_ok=True)
cmd=['npu-guard','run','--owner','baseline700-20261026/threehost_native_completion9','--wait','120','--timeout','300','--','bash',str(P/'launch.sh'),mode,op,cid]
start=time.time()
with (D/'guard.log').open('w') as f:
 child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 while child.poll() is None:
  try: meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
  except Exception:meta={}
  try:phase=json.loads((D/'result/PHASE.json').read_text())['phase']
  except Exception:phase=meta.get('phase','queue') if meta.get('pid')==child.pid else 'queue'
  hb={'at':time.time(),'stage':phase,'case':cid,'elapsed':time.time()-start,'guard_metadata':meta}
  (P/'HEARTBEAT.json').write_text(json.dumps(hb,indent=2))
  with (P/'HEARTBEAT.jsonl').open('a') as h:h.write(json.dumps(hb)+'\n')
  print(phase,cid,flush=True);time.sleep(10)
rc=child.wait();(D/'GUARD_FINAL.json').write_text(json.dumps({'command':cmd,'exit_code':rc,'start':start,'end':time.time(),'wall_seconds':time.time()-start},indent=2));print('EXIT',rc,flush=True)
raise SystemExit(rc)
