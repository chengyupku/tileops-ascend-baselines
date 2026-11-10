from control import *
import tarfile,hashlib
hb('seal_and_collect')
seal=f'''from pathlib import Path
import json,hashlib
p=Path({R!r});c=p/'square-4k-bfloat16'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
cap=json.loads((c/'CAPTURE.json').read_text());checks={{}}
for n,entry in cap.items():
 if entry.get('stored'):checks[n]=dict(actual_sha256=sha(c/(n+'.bin')),expected_sha256=entry['sha256'],bytes=(c/(n+'.bin')).stat().st_size)
assert all(x['actual_sha256']==x['expected_sha256'] for x in checks.values())
(p/'INPUTHASH_VERIFICATION.json').write_text(json.dumps(checks,indent=2))
exe=json.loads((c/'EXECUTION.json').read_text());mapped={{f:dict(sha256=sha(Path(f)),bytes=Path(f).stat().st_size) for f in exe['loaded_files'] if Path(f).is_file() and ('.so' in f or '/private_provider/' in f)}}
(p/'NATIVE_CLOSURE.json').write_text(json.dumps(mapped,indent=2))
files=[f for f in p.iterdir() if f.is_file()]+[f for f in c.iterdir() if f.is_file()]
pins={{str(f.relative_to(p)):sha(f) for f in files if f.name!='SEALED_HASHES.json'}}
(p/'SEALED_HASHES.json').write_text(json.dumps(pins,indent=2));print(json.dumps(dict(file_count=len(pins),inputs_verified=len(checks))))
'''
r=py(seal);(P/'SEAL_RECEIPT.json').write_text(json.dumps(r,indent=2));assert r['rc']==0,r
cmd=f'cd {R} && tar --exclude=./TileOPs --exclude=./tools --exclude=./private_provider --exclude=./square-4k-bfloat16/ASCEND_CACHE_PATH --exclude=./square-4k-bfloat16/ASCEND_WORK_PATH --exclude=./square-4k-bfloat16/XDG_CACHE_HOME --exclude=./square-4k-bfloat16/TORCH_EXTENSIONS_DIR --exclude=./square-4k-bfloat16/ASCEND_PROCESS_LOG_PATH --exclude=./square-4k-bfloat16/TRITON_CACHE_DIR --exclude=./square-4k-bfloat16/TMPDIR --exclude=./square-4k-bfloat16/TMP --exclude=./square-4k-bfloat16/TEMP --exclude=./square-4k-bfloat16/ACL_OP_COMPILER_CACHE_DIR -czf - .'
with (P/'server1_evidence.tar.gz').open('wb') as out, (P/'collection.log').open('w') as err:
 p=subprocess.Popen([SSH,HOST,cmd],stdout=out,stderr=err)
 while True:
  hb('collect_exact_bytes')
  try:rc=p.wait(timeout=25);break
  except subprocess.TimeoutExpired:pass
 assert rc==0,rc
D=P/'server1';D.mkdir()
with tarfile.open(P/'server1_evidence.tar.gz') as t:t.extractall(D,filter='data')
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
pins=json.loads((D/'SEALED_HASHES.json').read_text());assert all(sha(D/f)==h for f,h in pins.items())
(P/'LOCAL_VERIFICATION.json').write_text(json.dumps(dict(verified_files=len(pins),all_hashes_match=True),indent=2))
hb('evidence_verified_local');print('verified',len(pins))
