import subprocess,shlex,json,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent;ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code="""import json,pathlib,datetime
m=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text());p=m['pid'];a=pathlib.Path('/proc/'+str(p)+'/cmdline').read_bytes().decode().split('\\0');timeout=float(a[a.index('--timeout')+1]) if '--timeout' in a else None;elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(m['updated_at'])).total_seconds();print(json.dumps(dict(owner=m.get('owner'),pid=p,phase=m['phase'],timeout=timeout,since_phase_s=elapsed,remaining_s=timeout-elapsed if timeout else None)))"""
def query(item):
 n,h=item;r=subprocess.run([ssh,h,'python3 -I -S -B -c '+shlex.quote(code)],capture_output=True,text=True);return n,dict(returncode=r.returncode,output=r.stdout,error=r.stderr)
with concurrent.futures.ThreadPoolExecutor() as e:d=dict(e.map(query,{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}.items()))
(P/'HOST_RUNTIME_FORECAST.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
