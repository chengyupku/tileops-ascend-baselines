from pathlib import Path
import subprocess,json,time,datetime,concurrent.futures,sys
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11';ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';cid,attempt=sys.argv[1:];hosts={'server2':'openlibing-DevEnv-055409','server1':'openlibing-DevEnv-661495'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def probe(item):
 name,h=item;r=subprocess.run([ssh,h,'npu-guard status --json'],capture_output=True,text=True,timeout=30)
 try:d=json.loads(r.stdout)
 except ValueError:d={'state':'unknown','error':r.stderr,'raw':r.stdout}
 return name,d
start=time.time();chosen=None
while time.time()-start<900:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:states=dict(e.map(probe,hosts.items()))
 event=dict(at=now(),key=['ops-nn','GroupedGemmFwdOp',cid],states=states)
 with (P/'REMOTE_AVAILABILITY.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 for name in hosts:
  d=states[name]
  if d.get('state')=='idle' and not d.get('reservation',{}).get('locked'):chosen=name;break
 (P/'HEARTBEAT.json').write_text(json.dumps(dict(phase='waiting_remote_availability',currentkey=event['key'],host=chosen,lastprogress=now(),blockedreason=None if chosen else 'both compatible remote devices occupied; no numerical failure',resource_wait_s=time.time()-start),indent=2))
 if chosen:break
 time.sleep(30)
if chosen is None:raise SystemExit('No remote availability within900s; no device command submitted')
cmd=[ssh,hosts[chosen],f'python3 -B {R}/launch_remote.py {cid} {attempt}']
(P/(attempt+'_ROUTING.json')).write_text(json.dumps(dict(host=chosen,ssh_host=hosts[chosen],resource_wait_s=time.time()-start,command=cmd),indent=2))
with (P/(attempt+'_transport.log')).open('w') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  r=subprocess.run([ssh,hosts[chosen],'cat '+R+'/HEARTBEAT.json'],capture_output=True,text=True,timeout=30)
  try:d=json.loads(r.stdout)
  except ValueError:d={'remote_output':r.stdout}
  d.update(host=chosen,lastprogress=now(),transport_attempt=attempt)
  (P/'HEARTBEAT.json').write_text(json.dumps(d,indent=2));time.sleep(20)
print(chosen,cid,'transport_rc',proc.returncode,flush=True)
# Collect evidence only; do not claim numerical success from transport completion.
with (P/(attempt+'_receipts.tar')).open('wb') as f:
 subprocess.run([ssh,hosts[chosen],f'tar cf - --exclude=*.pt --exclude=ASCEND_PROCESS_LOG_PATH --exclude=.cache -C {R} {attempt} environment.sh private_python.py launch_remote.py run_remote.py'],stdout=f,check=False)
import tarfile
D=P/(chosen+'_collected');D.mkdir(exist_ok=True)
with tarfile.open(P/(attempt+'_receipts.tar')) as t:t.extractall(D,filter='data')
print('receipts',D/attempt,flush=True)
