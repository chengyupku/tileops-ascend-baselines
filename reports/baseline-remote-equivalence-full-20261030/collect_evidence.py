"""Read-only receipt and compiled-source collection; preserve full remote debug trees."""
import io,json,subprocess,tarfile,concurrent.futures,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent
hosts=[('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]
def collect(item):
 host,alias,root=item
 script='''import pathlib,tarfile,sys
h=pathlib.Path(ROOT)/'reports/baseline-remote-equivalence-full-20261030'/HOST
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
 for p in h.rglob('*'):
  if p.is_file() and (p.suffix=='.json' or p.name=='device.log' or p.name.endswith('_kernel.cpp') or p.name.endswith('_tiling_data.h')):
   t.add(p,arcname=str(p.relative_to(h)),recursive=False)
'''.replace('ROOT',repr(root)).replace('HOST',repr(host))
 result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',alias,'python3 -'],input=script.encode(),capture_output=True,check=True)
 dest=C/('evidence-'+host);dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(result.stdout)) as t:
  for m in t.getmembers():
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  t.extractall(dest,filter='data')
 files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
 (C/(host+'.evidence-sha256.json')).write_text(json.dumps(files,indent=2))
 return host,len(files)
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for x in pool.map(collect,hosts):print(x)
