import subprocess,json
from pathlib import Path
code=r'''
from pathlib import Path
import json,shutil,hashlib,time
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1');b=h/'native_build/selection_v1';receipt=json.loads((b/'prepermute_build_receipt.json').read_text());assert receipt['returncode']==0
src=b/'source/build/_CPack_Packages/Linux/External/cann-ops-transformer-prepermutepairs_linux-aarch64.run/packages/vendors/prepermutepairs_transformer'
dst=h/'deployments/ops_prepermute/vendors/prepermutepairs_transformer';assert not dst.exists();shutil.copytree(src,dst,symlinks=True)
rows=[];elf=[]
for p in src.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(src);data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();assert hashlib.sha256((dst/rel).read_bytes()).hexdigest()==sha
 rows.append({'path':str(rel),'bytes':len(data),'sha256':sha})
 if data[:4]==b'\x7fELF':
  machine=int.from_bytes(data[18:20],'little');assert machine==183;elf.append(str(rel))
fake=h/'deployments/ops_prepermute/root/.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer';fake.parent.mkdir(parents=True,exist_ok=True);fake.symlink_to(dst)
print(json.dumps({'at':time.time(),'build':receipt,'files':rows,'arm_elf':elf,'vendor':str(dst),'count':len(rows)}))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','python3 -'],input=code,text=True,capture_output=True,check=True);x=json.loads(r.stdout);Path(__file__).with_name('PREPERMUTE_ARM_DEPLOYMENT.json').write_text(json.dumps(x,indent=2));print({'count':x['count'],'arm_elf':x['arm_elf']})
