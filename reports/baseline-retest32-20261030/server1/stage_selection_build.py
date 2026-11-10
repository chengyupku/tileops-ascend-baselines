from pathlib import Path
import tarfile,subprocess,json,time,hashlib
r=Path.cwd();h=r/'reports/baseline-retest32-20261030/server1';src=r/'ascend_baselines/ops-transformer';archive=h/'archives/selection-build-source.tar.gz';manifest=[]
exclude={'.git','build','build_out','output','__pycache__'}
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(src.rglob('*')):
  rel=p.relative_to(src)
  if any(x in exclude for x in rel.parts):continue
  if p.is_file() and not p.is_symlink():
   b=p.read_bytes();manifest.append({'path':str(rel),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});tar.add(p,arcname=str(rel),recursive=False)
remote='/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1/native_build/selection_v1/source'
ssh=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
code='import shutil;from pathlib import Path;p=Path('+repr(remote)+');assert not p.exists();assert shutil.disk_usage(p.parents[3]).free>'+str(51*2**30+sum(x['bytes'] for x in manifest))+';p.mkdir(parents=True)'
subprocess.run(ssh+['python3 -'],input=code,text=True,check=True)
with archive.open('rb') as f:subprocess.run(ssh+['tar -xzf - -C '+remote],stdin=f,check=True)
code='manifest='+repr(manifest)+'\nroot='+repr(remote)+'''\nimport hashlib,json
from pathlib import Path
for x in manifest:assert hashlib.sha256((Path(root)/x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
print(json.dumps({'verified':len(manifest)}))
'''
subprocess.run(ssh+['python3 -'],input=code,text=True,check=True)
(h/'SELECTION_BUILD_SOURCE.json').write_text(json.dumps({'at':time.time(),'remote':remote,'archive':str(archive),'files':manifest},indent=2)+'\n')
print({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest)})
