import pathlib,subprocess,json,time,datetime,concurrent.futures,sys
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
op,cid=sys.argv[1:];key=['tilelang-ascend',op,cid];assert key in [r['key'] for r in json.loads((P/'SCOPE.json').read_text())]
out=P/'server2_runs'/(cid+'-legacy-pto-a5');out.mkdir(parents=True,exist_ok=True);assert not (out/'guard.json').exists()
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30);rec=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),argv=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr);(out/(h+'_status.json')).write_text(json.dumps(rec,indent=2));return h,rec
with concurrent.futures.ThreadPoolExecutor(3) as ex:
 statuses=dict(ex.map(status,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker05'
cmd=[SSH,'openlibing-DevEnv-055409','python3 '+remote+'/launch_remote_a5.py '+op+' '+cid]
print(subprocess.run(cmd).returncode,flush=True)
