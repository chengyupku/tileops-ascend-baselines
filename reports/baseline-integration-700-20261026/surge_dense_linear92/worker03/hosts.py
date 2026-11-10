import subprocess,json,datetime,sys
from pathlib import Path
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
from concurrent.futures import ThreadPoolExecutor
commands=[['npu-guard','status','--json']]+[[SSH,h,'npu-guard status --json'] for h in ['openlibing-DevEnv-661495','openlibing-DevEnv-055409']]
def run(cmd):
 r=subprocess.run(cmd,capture_output=True,text=True);return dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
r=list(ThreadPoolExecutor(3).map(run,commands));(P/('HOSTS_'+sys.argv[1]+'.json')).write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hosts=r),indent=2))
for i,x in enumerate(r):
 try: d=json.loads(x['stdout']);print(i,d['state'],d.get('reservation',{}).get('metadata',{}).get('owner'))
 except Exception:print(i,x['returncode'])
