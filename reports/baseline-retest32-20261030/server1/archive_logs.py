"""Archive only completed own attempt logs; verify local and remote gzip before unlink."""
import base64,gzip,hashlib,json,subprocess,time
from pathlib import Path
H=Path(__file__).resolve().parent
SSH=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','nice -n 15 ionice -c 3 python3 -']
select=r'''
import base64,gzip,hashlib,json
from pathlib import Path
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1')
active=json.loads((h/'ACTIVE.json').read_text()).get('batch');result=[]
for p in (h/'batches').glob('*/*/*.log'):
 if str(p.parents[1])==active:continue
 guard=p.parent/'guard.json'
 if not guard.exists() or json.loads(guard.read_text()).get('phase')!='finished':continue
 raw=p.read_bytes();packed=gzip.compress(raw,mtime=0)
 if len(packed)>=len(raw):continue
 result.append({'relative':str(p.relative_to(h)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'gzip':base64.b64encode(packed).decode()})
print(json.dumps(result))
'''
r=subprocess.run(SSH,input=select,text=True,capture_output=True,check=True);items=json.loads(r.stdout)
for row in items:
 packed=base64.b64decode(row['gzip']);assert hashlib.sha256(gzip.decompress(packed)).hexdigest()==row['sha256']
 p=H/'archived_logs'/(row['relative']+'.gz');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(packed)
 assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==row['sha256']
commit='items='+repr(items)+r'''
import base64,gzip,hashlib,json,time
from pathlib import Path
h=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030/server1');active=json.loads((h/'ACTIVE.json').read_text()).get('batch');saved=0;done=[]
for row in items:
 p=h/row['relative'];assert p.resolve().is_relative_to(h.resolve()) and p.suffix=='.log'
 if str(p.parents[1])==active:continue
 assert json.loads((p.parent/'guard.json').read_text())['phase']=='finished'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 q=p.with_suffix('.log.gz');packed=base64.b64decode(row['gzip'])
 if q.exists():assert q.read_bytes()==packed
 else:q.write_bytes(packed)
 assert hashlib.sha256(gzip.decompress(q.read_bytes())).hexdigest()==row['sha256']
 p.unlink();saved+=row['bytes']-len(packed);done.append({k:v for k,v in row.items() if k!='gzip'})
receipt={'at':time.time(),'files':done,'saved_bytes':saved,'local_copy_verified_before_remote_commit':True}
with (h/'LOG_ARCHIVES.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
print(json.dumps(receipt))
'''
r=subprocess.run(SSH,input=commit,text=True,capture_output=True,check=True);receipt=json.loads(r.stdout)
with (H/'LOG_ARCHIVES.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
print(json.dumps({'files':len(receipt['files']),'saved_bytes':receipt['saved_bytes']}))
