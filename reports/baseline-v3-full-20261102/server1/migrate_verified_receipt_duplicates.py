"""Remove only remote receipt duplicates proven present in verified local full archives."""
import json,hashlib,subprocess,tarfile,time
from pathlib import Path
P=Path(__file__).resolve().parent
items=[]
for ap in P.glob('runs/*/*/ARCHIVE.json'):
 a=json.loads(ap.read_text())
 if not a.get('verified'):continue
 archive=Path(a['local'])
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==a['sha256']
 with tarfile.open(archive) as t:
  for member in t.getmembers():
   name=member.name.removeprefix('./')
   if name not in ('correctness.json','measurement.json','FILES.json','native_maps.txt','native_hashes.json','guard.json','phase.json','DEPLOYMENT.json','TASK_PLOGS.json','DISK_STATUS.json'):continue
   b=t.extractfile(member).read();items.append([a['remote']+'/'+name,hashlib.sha256(b).hexdigest()])
code='''import json,hashlib
from pathlib import Path
items=ITEMS
freed=0
for name,sha in items:
 p=Path(name)
 assert '/baseline-v3-full-20261102/server1/runs/' in str(p)
 if not p.exists():continue
 assert (p.parent/'ARCHIVED.json').exists()
 b=p.read_bytes()
 assert hashlib.sha256(b).hexdigest()==sha,str(p)
 freed+=len(b);p.unlink()
print(json.dumps({'freed_bytes':freed,'checked':len(items)}))
'''.replace('ITEMS',repr(items))
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);r.check_returncode();print(r.stdout)
(P/f'RECEIPT_MIGRATION_{time.time_ns()}.json').write_text(json.dumps({'at':time.time(),'items':items,'result':json.loads(r.stdout)},indent=2))
