from pathlib import Path
import subprocess,json,datetime,time,tarfile,io,shlex
P=Path(__file__).resolve().parent.parent; D=Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker01'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
H='openlibing-DevEnv-661495';key=['tilelang-ascend','MHADecodePagedWithKVCacheFwdOp','serving-b2-float16']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def heartbeat(phase):
 data=dict(utc=utc(),key=key,phase=phase,last_substantial_output=str(D),blocker=None,next='collect exact original checker',deadline='2026-10-27T05:31:00+00:00',sla_overdue=utc()>='2026-10-27T05:31:00',accepted_added=0)
 (P/'HEARTBEAT.json').write_text(json.dumps(data,indent=2))
 if data['sla_overdue'] and not (D/'SLA_053100_SNAPSHOT.json').exists():
  data['snapshot_observed_at']=utc();data['status_at_deadline']='guard request pending or active; checker result not collected';(D/'SLA_053100_SNAPSHOT.json').write_text(json.dumps(data,indent=2))
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for n in ['run.py','launch.py']:t.add(D/n,arcname=D.name+'/'+n)
z=subprocess.run([SSH,H,'tar -xf - -C '+R],input=b.getvalue(),capture_output=True);assert z.returncode==0,z.stderr
cmd=[SSH,H,'python3 '+R+'/'+D.name+'/launch.py '+key[1]+' '+key[2]]
(D/'transport_command.json').write_text(json.dumps(dict(utc=utc(),argv=cmd),indent=2))
with (D/'transport.log').open('x') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 while proc.poll() is None:heartbeat('foreground_guard_request_or_execution');time.sleep(5)
(D/'transport_exit.json').write_text(json.dumps(dict(utc=utc(),exit_code=proc.returncode)))
z=subprocess.run([SSH,H,'tar -cf - -C '+R+'/'+D.name+' runs'],capture_output=True)
assert z.returncode==0,z.stderr
with tarfile.open(fileobj=io.BytesIO(z.stdout)) as t:t.extractall(D,filter='data')
heartbeat('exact_attempt_collected')
print('collected',D,proc.returncode,flush=True)
