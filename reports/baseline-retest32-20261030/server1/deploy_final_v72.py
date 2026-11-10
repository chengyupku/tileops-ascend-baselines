from pathlib import Path
import json,subprocess,base64,hashlib,time
r=Path.cwd();h=Path(__file__).resolve().parent
s=(h/'runner_v72_staged.py').read_text().replace("t['key'][1] in ('MLAVarlenFwdOp','SSDStatePassingFwdOp','GeluTanhAndMulFwdOp')", "t['key'][1]=='MLAVarlenFwdOp'")
files={f'tools/baseline_retest32_server1_{n}':(r/f'tools/baseline_retest32_server1_{n}').read_bytes() for n in ('final_v72.py','entry_v72.py','python_v72.sh')};files['tools/run_baseline_retest32_server1.py']=s.encode()
code='files='+repr({k:base64.b64encode(v).decode() for k,v in files.items()})+r'''
from pathlib import Path
import json,fcntl,base64,hashlib,time
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';assert not Path('/proc/1314756').exists();assert not (h/'STOP').exists()
f=(h/'supervisor.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=r/'tools/run_baseline_retest32_server1.py';(h/'runner_v71_preserved.py').write_bytes(p.read_bytes())
for n,d in files.items():
 p=r/n;v=base64.b64decode(d);q=p.with_suffix(p.suffix+'.stage');q.write_bytes(v);q.replace(p);assert p.read_bytes()==v
print(json.dumps({'at':time.time(),'sha256':{n:hashlib.sha256(base64.b64decode(d)).hexdigest() for n,d in files.items()},'old_pid_absent':True,'lock_free':True}))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);(h/'FINAL_INTERFACE_DEPLOYMENT.json').write_text(p.stdout);(r/'tools/run_baseline_retest32_server1.py').write_text(s);print(p.stdout)
