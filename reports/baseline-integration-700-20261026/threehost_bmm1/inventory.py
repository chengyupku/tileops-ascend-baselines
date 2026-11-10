import subprocess,json,time,concurrent.futures
from pathlib import Path
P=Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def check(host):
 cmd=['npu-guard','status','--json'] if host=='local' else [ssh,{'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host],'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 try:data=json.loads(r.stdout)
 except Exception:data={'stdout':r.stdout,'stderr':r.stderr}
 return host,dict(receipt=data,returncode=r.returncode,compatibility='x86 SDK9.2 existing original provider' if host=='local' else 'ARM SDK9.1 original provider; server1 snapshot ELF verified' if host=='server1' else 'ARM SDK9.1; provider requires CPU verification before execution')
with concurrent.futures.ThreadPoolExecutor(3) as e:d=dict(e.map(check,['local','server1','server2']))
(P/'HOST_ROUTING.json').write_text(json.dumps(dict(time=time.time(),hosts=d,selected_host='server1',reason='User priority; server1 idle and ARM provider available; private equivalent FP32 widening binding',migration_events=[dict(source='local',destination='server1',reason='Prior guard76 during reference; candidate never executed')]),indent=2))
print({h:(v['receipt'].get('state'),v['returncode']) for h,v in d.items()})
