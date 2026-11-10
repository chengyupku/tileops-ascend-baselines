from pathlib import Path
import sys,os,json,subprocess,time,platform,datetime
P=Path(__file__).resolve().parent;op,cid,attempt=sys.argv[1:];key=['tilelang-ascend',op,cid]
assert key in [r['key'] for r in json.loads((P/'SCOPE.json').read_text())]
local=platform.machine()=='x86_64';out=P/('local_runs' if local else 'runs')/(cid+'-'+attempt);out.mkdir(parents=True,exist_ok=True);assert not (out/'guard.json').exists()
env=os.environ.copy();env['WORKER_ATTEMPT']=attempt;env['PYTHONDONTWRITEBYTECODE']='1'
runner=P/('run_local_a5.py' if local else 'run_remote_a5.py');python=str(P/'local-python') if local else '/opt/tileops-all-pairs/bin/allpairs-python'
cmd=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker05','--wait','120','--timeout',str(300 if op=='GQAPagedFwdOp' else 600),'--',python,str(runner),op,cid]
r=dict(key=key,argv=cmd,started=time.time(),requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),changed_condition='First actual original PTO/A5 execution of this exact key; previous reservation expiry had no child; host selected after current three-host status; may join cooperative queue')
(out/'COMMAND.json').write_text(json.dumps(r,indent=2))
with (out/'guard.log').open('x') as f:r['returncode']=subprocess.run(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
r.update(finished=time.time(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
