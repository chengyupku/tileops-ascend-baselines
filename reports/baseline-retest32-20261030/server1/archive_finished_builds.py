"""Move completed private native build copies to a locally verified archive."""
import subprocess,json,hashlib,tarfile,time
from pathlib import Path
H=Path(__file__).resolve().parent
SSH=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
ROOT='/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1/native_build'
NAMES=['recurrent','causal_conv1d','chunk_gated_delta_rule_bwd','original-source']
code='root='+repr(ROOT)+'\nnames='+repr(NAMES)+r'''
from pathlib import Path
import hashlib,json,os
r=Path(root)
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:c=p.read_bytes()
 except OSError:continue
 assert not (root.encode() in c and any(w in c for w in (b'cmake',b'cc1plus',b'g++'))),c[:200]
m={}
for n in names:
 for p in (r/n).rglob('*'):
  if p.is_symlink():m[str(p.relative_to(r))]={'link':os.readlink(p)}
  elif p.is_file():m[str(p.relative_to(r))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(m))
'''
r=subprocess.run(SSH+['python3 -'],input=code,text=True,capture_output=True,check=True);manifest=json.loads(r.stdout)
assert manifest
out=H/'archives'/('native-build-'+str(time.time_ns())+'.tar.gz');out.parent.mkdir(exist_ok=True)
with out.open('wb') as f:subprocess.run(SSH+['tar -czf - -C '+ROOT+' '+' '.join(NAMES)],stdout=f,check=True)
with tarfile.open(out,'r:gz') as t:
 for rel,rec in manifest.items():
  member=t.getmember(rel)
  if 'link' in rec:assert member.issym() and member.linkname==rec['link']
  else:assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==rec['sha256'],rel
receipt={'at':time.time(),'archive_local':str(out),'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(manifest),'original_bytes':sum(x.get('bytes',0) for x in manifest.values()),'manifest':manifest}
out.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2))
commit='root='+repr(ROOT)+'\nnames='+repr(NAMES)+'\nexpected='+repr(manifest)+'\nreceipt='+repr({k:v for k,v in receipt.items() if k!='manifest'})+r'''
from pathlib import Path
import hashlib,json,shutil,os
r=Path(root)
for rel,rec in expected.items():
 p=r/rel
 if 'link' in rec:assert p.is_symlink() and os.readlink(p)==rec['link']
 else:assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],rel
# Copies only. Deployed runtime packages, original sources, caches, and logs remain.
(r/'ARCHIVED_BUILD_COPIES.json').write_text(json.dumps(receipt,indent=2))
for n in names:
 p=r/n;assert p.parent==r and p.is_dir() and not p.is_symlink();shutil.rmtree(p)
print(json.dumps(receipt))
'''
r=subprocess.run(SSH+['python3 -'],input=commit,text=True,capture_output=True,check=True);print(r.stdout)
