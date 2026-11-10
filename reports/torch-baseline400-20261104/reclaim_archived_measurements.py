"""Reclaim only verified archived campaign measurement JSON copies; retain local full raw."""
import pathlib,json,hashlib,subprocess,time,fcntl,shlex
P=pathlib.Path(__file__).resolve().parent
R='/opt/torch-baseline400-20261104/server1/runs'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run():
 with (P/'measurement-reclaim.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rows=[]
  for f in (P/'server1/runs').glob('*/a*/ARCHIVE.json'):
   d=f.parent
   if (d/'MEASUREMENT_RECLAIM.json').exists():continue
   a=json.loads(f.read_text());m=d/'measurement.json'
   if not a.get('accepted') or not a.get('verified') or not a.get('cleaned') or not m.exists():continue
   files=json.loads((d/'FILES.json').read_text())['files'];meta=files['measurement.json']
   assert m.stat().st_size==meta['size'] and sha(m)==meta['sha256']
   archive=pathlib.Path(a['archive']);assert archive.is_relative_to(P) and sha(archive)==a['sha256']
   rows.append({'rel':str(d.relative_to(P/'server1/runs')),'measurement':meta,'archive_sha256':a['sha256']})
  if not rows:return
  code='''import pathlib,json,hashlib,sys,time
root=pathlib.Path('/opt/torch-baseline400-20261104/server1/runs');result=[]
for item in json.load(sys.stdin):
 d=root/item['rel'];assert d.resolve().is_relative_to(root)
 a=json.loads((d/'ARCHIVED.json').read_text())
 if not (a.get('verified') and a.get('accepted') and a.get('sha256')==item['archive_sha256']):
  result.append(dict(item,at=time.time(),remote_measurement_removed=False,reason='remote archival receipt differs; retained for review'));continue
 assert json.loads((d/'DONE.json').read_text())['status']=='completed'
 m=d/'measurement.json';e=item['measurement']
 if m.exists():
  assert m.stat().st_size==e['size']
  with m.open('rb') as f:
   digest=hashlib.sha256()
   for block in iter(lambda:f.read(1048576),b''):digest.update(block)
   assert digest.hexdigest()==e['sha256']
  spool=m.resolve() if m.is_symlink() else None
  if spool is not None:assert spool.is_relative_to(pathlib.Path('/dev/shm/torch400-receipt-spool'))
  m.unlink()
  if spool is not None:spool.unlink()
 receipt=dict(item,at=time.time(),local_full_archive_verified=True,remote_measurement_removed=True)
 (d/'MEASUREMENT_RECLAIM.json').write_text(json.dumps(receipt));result.append(receipt)
print(json.dumps(result))'''
  response=subprocess.run(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495',shlex.join(['python3','-c',code])],input=json.dumps(rows),text=True,capture_output=True,check=False,timeout=600)
  if response.returncode:
   (P/'server1/MEASUREMENT_RECLAIM_ERROR.json').write_text(json.dumps({'at':time.time(),'stderr':response.stderr,'stdout':response.stdout}))
   raise RuntimeError(response.stderr)
  results=json.loads(response.stdout)
  for r in results:(P/'server1/runs'/r['rel']/'MEASUREMENT_RECLAIM.json').write_text(json.dumps(r,indent=2))
  print(json.dumps({'at':time.time(),'count':len(results),'reclaimed_GiB':sum(r['measurement']['size'] for r in results if r.get('remote_measurement_removed'))/2**30}),flush=True)
if __name__=='__main__':run()
