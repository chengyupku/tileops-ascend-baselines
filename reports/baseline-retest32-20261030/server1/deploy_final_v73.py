from pathlib import Path
import subprocess,base64,json,time
h=Path(__file__).resolve().parent;s=(h/'runner_v73_staged.py').read_text();code='data='+repr(base64.b64encode(s.encode()).decode())+r'''
from pathlib import Path
import fcntl,json,base64,hashlib,time
r=Path('/opt/tileops-baseline-cache/workspace');h=r/'reports/baseline-retest32-20261030/server1';assert not Path('/proc/1347413').exists()
f=(h/'supervisor.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
stop=json.loads((h/'STOP').read_text());assert stop['reason'].startswith('Drain v72 after current')
p=r/'tools/run_baseline_retest32_server1.py';(h/'runner_v72_preserved.py').write_bytes(p.read_bytes());b=base64.b64decode(data);q=p.with_suffix('.stage');q.write_bytes(b);q.replace(p);(h/'STOP_v72_drain.json').write_text(json.dumps(stop));(h/'STOP').unlink()
print(json.dumps({'at':time.time(),'old_pid_absent':True,'lock_free':True,'sha256':hashlib.sha256(b).hexdigest()}))
'''
p=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);(h/'FINAL_V73_DEPLOYMENT.json').write_text(p.stdout);Path('tools/run_baseline_retest32_server1.py').write_text(s);print(p.stdout)
