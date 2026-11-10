from pathlib import Path
import subprocess,concurrent.futures,json,sys,datetime
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
label=sys.argv[1]
def check(host):
 cmd=['npu-guard','status','--json'] if host=='local' else [SSH,host,'LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver python3 /opt/tileops-all-pairs/workspace/tools/npu_guard.py status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
 (P/(label+'_'+host+'_STATUS.json')).write_text(r.stdout)
 d=json.loads(r.stdout);return dict(host=host,rc=r.returncode,state=d['state'],pids=d.get('pids'),reservation=d.get('reservation'),arch='x86_64' if host=='local' else 'aarch64',compatible=host=='local',reason='verified original x86 recipe' if host=='local' else 'original device package key24578 missing; build required')
with concurrent.futures.ThreadPoolExecutor() as ex: rows=list(ex.map(check,['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409']))
(P/(label+'_HOST_ROUTING.json')).write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=rows),indent=2))
print([(r['host'],r['state'],r['pids']) for r in rows])
