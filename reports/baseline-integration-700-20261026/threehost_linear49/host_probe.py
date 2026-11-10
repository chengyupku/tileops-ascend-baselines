import subprocess,json,pathlib,datetime
r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);s=json.loads(r.stdout);m=s.get('reservation',{}).get('metadata',{});p=pathlib.Path('/proc/'+str(m.get('pid',0))+'/cmdline');timeout=None
if p.is_file():
 a=p.read_bytes().decode(errors='replace').split('\0')
 if '--timeout' in a:timeout=a[a.index('--timeout')+1]
s['current_guard_timeout_seconds']=timeout
print(json.dumps(s))
