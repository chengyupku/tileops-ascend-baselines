import subprocess,json,concurrent.futures
from pathlib import Path
C=Path(__file__).resolve().parent
hosts=[('server1','openlibing-DevEnv-661495','/opt/tileops-baseline-cache/workspace'),('server2','openlibing-DevEnv-055409','/opt/tileops-all-pairs/workspace')]
def get(item):
 h,a,r=item
 script='''import pathlib,json
root=pathlib.Path(ROOT);h=root/'reports/baseline-remote-equivalence-20261030'/HOST
out={}
for p in h.rglob('correctness.json'):
 row=json.loads(p.read_text()); out[str(p.relative_to(h))]={'correctness':row}
 for name in ['guard.json','device.log']:
  q=p.with_name(name)
  if q.exists():out[str(p.relative_to(h))][name]=json.loads(q.read_text()) if name.endswith('.json') else q.read_text(errors='replace')[-20000:]
print(json.dumps(out))
'''.replace('ROOT',repr(r)).replace('HOST',repr(h))
 v=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes',a,'python3 -'],input=script,capture_output=True,text=True,check=True)
 (C/(h+'.evidence.json')).write_text(v.stdout);return h,len(json.loads(v.stdout))
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 for v in pool.map(get,hosts):print(v)
