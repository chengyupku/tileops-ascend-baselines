import json,subprocess,io,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent
x=json.load(open(H/'INSPECTION.latest.json'))
paths=[]
for r in x['pending_review']:
 p=Path(r['measurement']);paths.extend([str(p),str(p.with_name('correctness.json')),r['guard']])
script="import pathlib,sys,tarfile\npaths="+repr(paths)+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:\n for name in paths:\n  p=pathlib.Path(name)\n  if p.is_file():t.add(p,arcname=str(p.relative_to('/opt/tileops-all-pairs/workspace')),recursive=False)\n"
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-055409','python3 -'],input=script.encode(),capture_output=True,check=True)
dest=H/'pending_evidence';dest.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(dest,filter='data')
print('Collected pending checker, measurement and guard receipts')
