import pathlib,subprocess,json,time,concurrent.futures,sys,shlex
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';hosts={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}
def status(h):
 cmd=['npu-guard','status','--json'] if h=='local' else [str(ssh),hosts[h],'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True);return dict(host=h,at=time.time(),rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
with concurrent.futures.ThreadPoolExecutor() as e:states=list(e.map(status,['server1','server2','local']))
f=P/('routing_'+str(time.time_ns())+'.json');f.write_text(json.dumps(states,indent=2));print([(s['host'],s['rc']) for s in states])
r=json.loads((P/'HOST_ROUTING.json').read_text());r['latest_status']=states;r['routing_receipts']=r.get('routing_receipts',[])+[str(f)];(P/'HOST_ROUTING.json').write_text(json.dumps(r,indent=2))
