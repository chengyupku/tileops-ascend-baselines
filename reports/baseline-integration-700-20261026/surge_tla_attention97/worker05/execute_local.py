import pathlib,subprocess,json,time,datetime,concurrent.futures,sys
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
op,cid=sys.argv[1:];key=['tilelang-ascend',op,cid];assert key in [r['key'] for r in json.loads((P/'SCOPE.json').read_text())]
out=P/'local_runs'/(cid+'-original-default');out.mkdir(parents=True,exist_ok=True);assert not (out/'guard.json').exists()
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);rec=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr);(out/(h+'_status.json')).write_text(json.dumps(rec,indent=2));return h,rec
with concurrent.futures.ThreadPoolExecutor(3) as ex:
 statuses=dict(ex.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
cmd=['npu-guard','run','--owner','baseline700-20261026/surge_tla_attention97/worker05','--wait','120','--timeout','600','--',str(P/'local-python'),str(P/'run_local.py'),op,cid]
rec=dict(key=key,argv=cmd,requested_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),started=time.time(),changed_condition='server1 queue expired without child; migrate to compatible x86 original-target recipe')
(out/'COMMAND.json').write_text(json.dumps(rec,indent=2))
with (out/'guard.log').open('x') as f:r=subprocess.run(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT)
rec.update(returncode=r.returncode,finished=time.time(),end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
