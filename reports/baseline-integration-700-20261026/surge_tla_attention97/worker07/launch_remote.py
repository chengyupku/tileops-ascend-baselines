from pathlib import Path
import subprocess,json,os,time,sys,datetime,concurrent.futures
P=Path(__file__).resolve().parent; W=P.parents[3];op,cid=sys.argv[1:]; key=['tilelang-ascend',op,cid]
assert key in [x['key'] for x in json.loads((P/'SCOPE.json').read_text())]
out=P/'runs'/(cid+'-pto-A5-original-branch');out.mkdir(parents=True,exist_ok=True)
assert not (out/'guard.json').exists(),'no unchanged retries'
ssh=W/'.unused'
def status(host):
 cmd=['npu-guard','status','--json'] if host=='local' else [str(ssh),host,'npu-guard status --json']
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
 return dict(host=host,argv=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr,received_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
# Three-host status saved by calling supervisor immediately before ssh submission.
env=os.environ.copy();env['INTEGRATION700_TLA_ROUTE']='legacy'; env['PYTHONDONTWRITEBYTECODE']='1'
cmd=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker07','--wait','120','--timeout','600','--','/opt/tileops-all-pairs/bin/allpairs-python',str(P/'run_remote.py'),op,cid]
r=dict(key=key,argv=cmd,requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),start=time.time(),host=__import__('socket').gethostname())
(out/'command.json').write_text(json.dumps(r,indent=2))
(P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=r['requested_at'],current_key=key,phase='guard_wait_or_device_run',last_output=str(out/'threehost_status.json'),blocker=None,next='original checker'),indent=2))
with (out/'guard.log').open('x') as f:r['returncode']=subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
r.update(end=time.time(),ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
