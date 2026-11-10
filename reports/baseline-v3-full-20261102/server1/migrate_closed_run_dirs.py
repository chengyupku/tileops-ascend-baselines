"""Reclaim closed remote task directories only after full local archive verification."""
import json,hashlib,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
items=[]
for f in P.glob('runs/*/*/ARCHIVE.json'):
 a=json.loads(f.read_text())
 if not a.get('verified'):continue
 assert hashlib.sha256(Path(a['local']).read_bytes()).hexdigest()==a['sha256']
 items.append([a['remote'],a['sha256'],json.loads((f.parent/'FILES.json').read_text())['files']])
code='''import json,hashlib,shutil
from pathlib import Path
items=ITEMS
count=0;freed=0
for name,sha,manifest in items:
 p=Path(name)
 assert '/baseline-v3-full-20261102/server1/runs/' in name
 if not p.exists():continue
 a=p/'ARCHIVED.json'
 if not a.exists():continue
 assert json.load(a.open())['sha256']==sha
 for f in p.rglob('*'):
  assert not f.is_symlink()
  if not f.is_file() or f==a:continue
  rel=str(f.relative_to(p))
  if rel=='FILES.json':continue
  assert rel in manifest,rel
  assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[rel]['sha256'],str(f)
  freed+=f.stat().st_size
 shutil.rmtree(p);count+=1
 try:p.parent.rmdir()
 except OSError:pass
print(json.dumps({'directories':count,'file_bytes':freed}))
'''.replace('ITEMS',repr(items))
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True);r.check_returncode();print(r.stdout)
(P/f'CLOSED_DIR_MIGRATION_{time.time_ns()}.json').write_text(json.dumps({'result':json.loads(r.stdout),'archives':[[x[0],x[1]] for x in items]},indent=2))
