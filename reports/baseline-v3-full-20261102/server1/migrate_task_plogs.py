"""Archive closed task-owned plogs; require guard PID and filename timestamp binding."""
import json,datetime,subprocess,hashlib,tarfile,time,shlex
from pathlib import Path
P=Path(__file__).resolve().parent
ssh=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
guards=[]
for f in P.glob('runs/*/*/guard.json'):
 g=json.loads(f.read_text());ap=f.parent/'ARCHIVE.json'
 if not ap.exists() or g.get('phase')!='finished':continue
 a=json.loads(ap.read_text())
 if not a.get('verified'):continue
 guards.append([g['child']['pid'],g['child']['start_ticks'],g['acquired_at'],g['updated_at']])
code='''from pathlib import Path
import datetime,json,hashlib
result={}
for pid,ticks,start,end in GUARDS:
 if Path('/proc/'+str(pid)).exists():continue
 lo=datetime.datetime.fromisoformat(start).timestamp()-5;hi=datetime.datetime.fromisoformat(end).timestamp()+5
 for p in Path('/root/ascend/log').rglob('plog-'+str(pid)+'_*.log'):
  try:at=datetime.datetime.strptime(p.stem.split('_',1)[1],'%Y%m%d%H%M%S%f').replace(tzinfo=datetime.timezone.utc).timestamp()
  except ValueError:continue
  if not lo<=at<=hi:continue
  result[str(p)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}
print(json.dumps(result))
'''.replace('GUARDS',repr(guards))
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True);manifest=json.loads(r.stdout)
if not manifest:print('no closed task logs');raise SystemExit
out=P/f'closed-plogs-{time.time_ns()}.tar.gz'
with out.open('wb') as f:subprocess.run(ssh+['tar -czf - -- '+ ' '.join(shlex.quote(n) for n in manifest)],stdout=f,check=True)
with tarfile.open(out) as t:
 for n,v in manifest.items():assert hashlib.sha256(t.extractfile(n.lstrip('/')).read()).hexdigest()==v['sha256'],n
code='''from pathlib import Path
import hashlib,json
m=MANIFEST
freed=0
for n,v in m.items():
 p=Path(n)
 assert str(p).startswith('/root/ascend/log/') and p.name.startswith('plog-')
 if not p.exists():continue
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'],n
 p.unlink();freed+=v['size']
print(json.dumps({'freed_bytes':freed}))
'''.replace('MANIFEST',repr(manifest))
r=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True)
out.with_suffix('.json').write_text(json.dumps({'archive':str(out),'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'manifest':manifest,'result':json.loads(r.stdout)},indent=2));print(r.stdout)
