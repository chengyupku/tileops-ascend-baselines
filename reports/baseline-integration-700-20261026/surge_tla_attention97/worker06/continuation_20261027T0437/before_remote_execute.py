from pathlib import Path
import os,subprocess,json,datetime,time,sys,concurrent.futures,shlex,tarfile
P=Path(__file__).resolve().parent;R=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker06')
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';OWNER='baseline700-20261026/surge_tla_attention97/worker06'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
host,op,cid=sys.argv[1:];key=['tilelang-ascend',op,cid];name=cid+os.environ.get('WORKER_ATTEMPT_SUFFIX','-pto-A5-original-branch');out=P/host/'runs'/name;out.mkdir(parents=True,exist_ok=True);remoteout=R/'runs'/name
assert not (out/'guard.json').exists()
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 (out/(h+'_status.json')).write_text(json.dumps(dict(collected_at=now(),command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2));return h,r.returncode
with concurrent.futures.ThreadPoolExecutor(3) as e:print(list(e.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409'])),flush=True)
# Keep every cache and driver output inside the private remote directory from interpreter startup.
env={n:str(remoteout/n) for n in ['TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','ASCEND_PROCESS_LOG_PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']};env['PYTHONDONTWRITEBYTECODE']='1';env['WORKER_ATTEMPT_SUFFIX']=os.environ.get('WORKER_ATTEMPT_SUFFIX','-pto-A5-original-branch')
cmd=['npu-guard','run','--owner',OWNER,'--wait','120','--timeout','180','--on-conflict','terminate','--','/opt/tileops-all-pairs/bin/allpairs-python',str(R/'run_case.py'),op,cid]
remote='mkdir -p '+ ' '.join(shlex.quote(v) for k,v in env.items() if k not in ['PYTHONDONTWRITEBYTECODE','WORKER_ATTEMPT_SUFFIX'])+' && cd '+shlex.quote(str(R))+' && env '+ ' '.join(shlex.quote(k+'='+v) for k,v in env.items())+' '+shlex.join(cmd)
d=dict(key=key,requested=now(),start=time.time(),command=[SSH,host,remote],guard_command=cmd,host=host,changed_condition='Exact key unrun on selected compatible ARM host; migration after local queue if applicable.')
(out/'command.json').write_text(json.dumps(d,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,phase='remote_guard_submitted',last_output=str(out/'command.json'),next='original checker'),indent=2))
with (out/'guard.log').open('x') as f:d['guard_exit']=subprocess.run([SSH,host,remote],stdout=f,stderr=subprocess.STDOUT).returncode
d.update(end=time.time(),ended=now());d['wall_seconds']=d['end']-d['start'];(out/'guard.json').write_text(json.dumps(d,indent=2))
# Original inputs remain in private remote case directory; collect other complete case evidence.
identity_code="from pathlib import Path;import hashlib,json;p=Path("+repr(str(remoteout/'inputs.pt'))+");r={'path':str(p),'exists':p.exists()};r.update({'size':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} if p.exists() else {});print(json.dumps(r))"
ident=subprocess.run([SSH,host,'/opt/tileops-ci/runtime/bin/python -I -S -B -c '+shlex.quote(identity_code)],capture_output=True,text=True)
(out/'INPUTS_IDENTITY.json').write_text(json.dumps(dict(exit=ident.returncode,stdout=ident.stdout,stderr=ident.stderr),indent=2))
with (out/'evidence.tar').open('wb') as f:r=subprocess.run([SSH,host,'tar --exclude=./inputs.pt -cf - -C '+shlex.quote(str(remoteout))+' .'],stdout=f,stderr=subprocess.PIPE)
if r.returncode==0:
 with tarfile.open(out/'evidence.tar') as t:t.extractall(out,filter='data')
row=dict(key=key,host=host,status='device_timeout' if d['guard_exit']==124 else 'not_started_guard_wait' if d['guard_exit']==75 else 'guard_visibility_or_external_failure' if d['guard_exit'] in [76,78] else 'checker_receipt_available' if (out/'checker.json').exists() else 'execution_failed' if d['guard_exit'] else 'requires_review',guard=str(out/'guard.json'),checker=str(out/'checker.json') if (out/'checker.json').exists() else None,native=str(out/'native.json') if (out/'native.json').exists() else None,accepted_added=0,registered=False,wall_seconds=d['wall_seconds'],collection_exit=r.returncode)
(out/'RESULT.json').write_text(json.dumps(row,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps(dict(at=now(),key=key,phase='case_finished',last_output=str(out/'RESULT.json'),next='review and next exact key'),indent=2));print(json.dumps(row),flush=True)
