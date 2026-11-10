"""User-authorized migration of Oct29 TLA launch evidence, preserving verified local archive."""
import subprocess,json,hashlib,tarfile,time,os,shlex
from pathlib import Path
H=Path(__file__).resolve().parent
SSH=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','-o','ConnectTimeout=10','openlibing-DevEnv-661495']
ROOT='/opt/tileops-baseline-cache/workspace/.cache/all-pairs/tla/launch-evidence'
REMOTE='/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1'
def run(code):return subprocess.run(SSH+['nice -n 15 ionice -c 3 python3 -'],input=code,text=True,capture_output=True,check=True).stdout
check=r'''
def unused(paths):
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   for f in (proc/'fd').iterdir():
    try:target=os.readlink(f)
    except FileNotFoundError:continue
    assert not any(target==x or target.startswith(x+'/') for x in paths),(str(f),target)
   maps=(proc/'maps').read_text(errors='replace')
   assert not any(x in maps for x in paths),str(proc)
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   assert not any(x in cmd for x in paths),str(proc)
  except (FileNotFoundError,ProcessLookupError):continue
'''
code='root='+repr(ROOT)+'\n'+r'''
from pathlib import Path
import json,hashlib,os,datetime
r=Path(root);selected=[]
for p in r.iterdir():
 if p.is_dir() and not p.is_symlink() and datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).date().isoformat()=='2026-10-29':selected.append(p)
'''+check+r'''
unused([str(x) for x in selected]);m={}
for d in selected:
 for p in d.rglob('*'):
  rel=str(p.relative_to(r))
  if p.is_symlink():m[rel]={'link':os.readlink(p)}
  elif p.is_file():
   st=p.stat();sha=hashlib.sha256(p.read_bytes()).hexdigest();assert (st.st_size,st.st_mtime_ns)==(p.stat().st_size,p.stat().st_mtime_ns)
   m[rel]={'sha256':sha,'bytes':st.st_size}
print(json.dumps({'directories':[p.name for p in selected],'manifest':m,'bytes':sum(v.get('bytes',0) for v in m.values())}))
'''
m=json.loads(run(code));assert m['directories'] and m['manifest'];out=H/'archives'/('historical-tla-launch-'+str(time.time_ns())+'.tar.gz');out.parent.mkdir(exist_ok=True)
print(json.dumps({'stage':'transfer','directories':len(m['directories']),'files':len(m['manifest']),'bytes':m['bytes'],'archive':str(out)}),flush=True)
with out.with_suffix('.partial').open('wb') as f:
 subprocess.run(SSH+['nice -n 15 ionice -c 3 tar -czf - -C '+shlex.quote(ROOT)+' --null -T -'],input=b''.join(x.encode()+b'\0' for x in m['directories']),stdout=f,check=True);f.flush();os.fsync(f.fileno())
out.with_suffix('.partial').replace(out)
with tarfile.open(out,'r:gz') as t:
 for rel,rec in m['manifest'].items():
  v=t.getmember(rel)
  if 'link' in rec:assert v.issym() and v.linkname==rec['link']
  else:assert hashlib.sha256(t.extractfile(v).read()).hexdigest()==rec['sha256'],rel
m.update(archive=str(out),archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),at=time.time(),authorization='User requested continuous migration overlapping CPU/NPU in direct response to historicalTLA migration proposal')
manifest=out.with_suffix('.manifest.json')
with manifest.open('w') as f:json.dump(m,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps({'stage':'local_verified','archive':str(out),'sha256':m['archive_sha256']}),flush=True)
code='root='+repr(ROOT)+'\nm='+repr(m)+'\nremote='+repr(REMOTE)+'\n'+r'''
from pathlib import Path
import os,json,hashlib,shutil
r=Path(root)
'''+check+r'''
unused([str(r/n) for n in m['directories']])
actual={str(p.relative_to(r)) for n in m['directories'] for p in (r/n).rglob('*') if p.is_file() or p.is_symlink()};assert actual==set(m['manifest'])
for rel,rec in m['manifest'].items():
 p=r/rel
 if 'link' in rec:assert p.is_symlink() and os.readlink(p)==rec['link']
 else:assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],rel
receipt={k:v for k,v in m.items() if k!='manifest'}
(Path(remote)/'HISTORICAL_TLA_MIGRATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
for n in m['directories']:
 p=r/n;assert p.parent==r and p.is_dir() and not p.is_symlink();shutil.rmtree(p)
print(json.dumps({'stage':'remote_committed','bytes_archived':m['bytes'],'free_bytes':shutil.disk_usage(r).free,'archive':m['archive']}))
'''
print(run(code),flush=True)
(H/'HISTORICAL_TLA_MIGRATION_RESULT.json').write_text(json.dumps({k:v for k,v in m.items() if k!='manifest'},indent=2)+'\n')
