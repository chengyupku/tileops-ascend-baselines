import subprocess,time,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent; case=sys.argv[1]; D=P/case
assert case in ['moe-dgrad-float16','moe-fwd-float16']
def dump(n,x): (D/n).write_text(json.dumps(x,indent=2)+'\n')
def snapshot(n):
 r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);(D/n).write_text(r.stdout);return r.returncode
print((P/'FEEDBACK.txt').read_text(),flush=True)
snapshot('GUARD_BEFORE.json');start=time.time()
command=['npu-guard','run','--owner','baseline700-20261026/fp16_grouped_budget900_2','--wait','600','--timeout','900','--','bash',str(P/'launch_one.sh'),case]
with (D/'full.log').open('w') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 last=0; prev=None
 while child.poll() is None:
  now=time.time()
  try: meta=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
  except Exception:meta={}
  if meta.get('pid')!=child.pid:meta={}
  try: phase=json.loads((D/'result/PHASE.json').read_text())['phase']
  except Exception:phase=meta.get('phase','queue')
  if phase!=prev:
   print('PHASE',phase,'FEEDBACK', (P/'FEEDBACK.txt').read_text(),flush=True);prev=phase
  if now-last>=50:
   hb={'at':now,'case':case,'phase':phase,'guard_pid':child.pid,'child':meta.get('child'),'elapsed':now-start}
   hb['last_progress']=phase
   (P/'HEARTBEAT.json').write_text(json.dumps(hb,indent=2)+'\n')
   with (P/'HEARTBEAT.jsonl').open('a') as f:f.write(json.dumps(hb)+'\n')
   print('HEARTBEAT',json.dumps(hb),flush=True);last=now
  time.sleep(5)
 rc=child.wait()
end=time.time();dump('GUARD_FINAL.json',{'exit_code':rc,'start':start,'end':end,'wall_seconds':end-start,'command':command,'metadata':json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())})
snapshot('GUARD_AFTER.json')
print('FINISHED',case,rc,flush=True)
raise SystemExit(rc)
