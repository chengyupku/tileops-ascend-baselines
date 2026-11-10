"""Independent CPU-only archive pass; never takes the device lock."""
import fcntl,json,subprocess,time,os
from pathlib import Path
H=Path(__file__).resolve().parent
lock=(H/'continuous_archive.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
started=time.time();steps=[]
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','nice -n 15 ionice -c 3 python3 -']
code=r'''
import json,time,re
from pathlib import Path
from datetime import datetime,timezone
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
windows={}
for p in (h/'batches').glob('*/*/guard.json'):
 try:g=json.loads(p.read_text())
 except (ValueError,FileNotFoundError):continue
 if g.get('phase')!='finished' or not g.get('owner','').startswith('baseline-retest32-20261030/server1/'):continue
 pid=g.get('child',{}).get('pid')
 if not pid or Path('/proc/'+str(pid)).exists():continue
 windows.setdefault(pid,[]).append((datetime.fromisoformat(g['requested_at']).timestamp()-5,datetime.fromisoformat(g['updated_at']).timestamp()+5))
files=[]
for p in Path('/root/ascend/log/run/plog').glob('plog-*.log'):
 m=re.match(r'plog-(\d+)_(\d{17})',p.name)
 if not m or p.is_symlink():continue
 pid=int(m[1])
 if pid not in windows:continue
 stamp=datetime.strptime(m[2],'%Y%m%d%H%M%S%f').replace(tzinfo=timezone.utc).timestamp()
 if any(a<=stamp<=b for a,b in windows[pid]):files.append({'path':str(p),'pid':pid,'bytes':p.stat().st_size})
r={'at':time.time(),'files':files,'total_bytes':sum(x['bytes'] for x in files)}
(h/'OWN_COMPLETED_PLOG_INVENTORY.json').write_text(json.dumps(r))
print(json.dumps({'files':len(files),'bytes':r['total_bytes']}))
'''
try:
 r=subprocess.run(ssh,input=code,text=True,capture_output=True,check=True,timeout=120)
 inventory=json.loads(r.stdout);steps.append({'inventory':inventory})
 scripts=['archive_logs.py']+(['archive_owned_plogs.py'] if inventory['files'] else [])
 for script in scripts:
  r=subprocess.run(['python3',str(H/script)],capture_output=True,text=True,timeout=600)
  steps.append({'script':script,'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if r.returncode:raise RuntimeError(script+' failed')
 status='ok'
except Exception as e:status='error';steps.append({'error':str(e)})
record={'started_at':started,'finished_at':time.time(),'status':status,'steps':steps,'independent_of_cpu_npu_queue':True}
p=H/'MIGRATION_STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(p)
with (H/'MIGRATION_HISTORY.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record))
raise SystemExit(status!='ok')
