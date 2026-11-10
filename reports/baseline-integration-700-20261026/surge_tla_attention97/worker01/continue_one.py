from pathlib import Path
import json,subprocess,time,datetime,concurrent.futures,tarfile,io,sys
P=Path(__file__).resolve().parent;D=P/'continuation_20261027T043900';R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker01'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
op,cid,host=sys.argv[1:];key=['tilelang-ascend',op,cid]
assert key in [x['key'] for x in json.loads((P/'CONTINUATION_SCOPE.json').read_text())]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hb(phase,**kw):(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=utc(),phase=phase,key=key,**kw),indent=2))
def status(h):
 c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 z=subprocess.run(c,capture_output=True,text=True,timeout=40);return dict(host=h,observed_at=utc(),rc=z.returncode,stdout=z.stdout,stderr=z.stderr)
with concurrent.futures.ThreadPoolExecutor() as e:ss=list(e.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
(D/(cid+'.'+host+'.admission.json')).write_text(json.dumps(ss,indent=2))
cmd=['python3',str(P/'launch_local_continuation.py'),op,cid] if host=='local' else [SSH,host,f'python3 {R}/launch_continuation.py {op} {cid}']
print(utc(),cmd,flush=True)
with (D/(cid+'.'+host+'.launch.log')).open('x') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  hb('continuation_foreground_guard',host=host,subprocess_pid=proc.pid,last_output=str(D/(cid+'.'+host+'.admission.json')),next='collect complete original checker and guard receipt');time.sleep(20)
if host=='local':out=D/'local_runs'/(cid+'-pto-A5')
else:
 z=subprocess.run([SSH,host,f'tar -cf - -C {R}/continuation_20261027T043900 runs/{cid}-pto-A5'],capture_output=True)
 dest=D/host;dest.mkdir(exist_ok=True)
 if z.returncode==0:
  with tarfile.open(fileobj=io.BytesIO(z.stdout)) as t:t.extractall(dest,filter='data')
 out=dest/'runs'/(cid+'-pto-A5')
g=json.loads((out/'guard.json').read_text()) if (out/'guard.json').exists() else {};c=json.loads((out/'checker.json').read_text()) if (out/'checker.json').exists() else None
r=dict(key=key,host=host,utc=utc(),transport_exit=proc.returncode,guard=g,checker=c,evidence=str(out),accepted_added=0,numerical_success=False,runtime_ready=False,status='not_started_guard75' if g.get('returncode')==75 else 'exact_attempt_pending_review')
(D/(cid+'.'+host+'.RESULT.json')).write_text(json.dumps(r,indent=2));hb('continuation_attempt_collected',last_output=str(D/(cid+'.'+host+'.RESULT.json')),next='review exact failure/success without sibling substitution')
print(utc(),'END',g.get('returncode'),c.get('status') if c else None,flush=True)
