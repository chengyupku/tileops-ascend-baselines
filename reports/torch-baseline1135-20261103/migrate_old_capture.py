import subprocess,pathlib,json,hashlib,gzip,shlex,time
P=pathlib.Path(__file__).resolve().parent;S=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495'];base='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_fp16_remaining11/moe01';out=P/'migration';out.mkdir(exist_ok=True)
for name in ['reference_0_input1.bin','reference_0_post_input1.bin','candidate_0_input1.bin','candidate_0_post_input1.bin']:
 remote=base+'/'+name;dst=out/(name+'.gz');receipt=out/(name+'.json')
 if receipt.exists() and json.loads(receipt.read_text()).get('removed'):continue
 code="import pathlib,hashlib,json; p=pathlib.Path("+repr(remote)+"); s=p.stat(); print(json.dumps({'size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}))"
 meta=json.loads(subprocess.check_output(S+['/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],text=True));print('hashed',name,flush=True)
 with dst.with_suffix('.partial').open('wb') as f:subprocess.run(S+['gzip -1 -c -- '+shlex.quote(remote)],stdout=f,check=True)
 dst.with_suffix('.partial').replace(dst)
 with gzip.open(dst,'rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
 assert actual==meta['sha256'];meta.update(remote=remote,archive=str(dst),verified=True,at=time.time());receipt.write_text(json.dumps(meta,indent=2))
 # Only a stopped historical capture payload; never cache/native packages or memory.
 code="""import pathlib,hashlib,json,os
p=pathlib.Path(PATH);m=META
assert p.stat().st_size==m['size'] and p.stat().st_mtime_ns==m['mtime_ns']
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try: assert fd.resolve()!=p, 'capture still open'
   except FileNotFoundError: pass
 except PermissionError: raise RuntimeError('cannot verify open handles')
assert hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==m['sha256']
p.unlink()
print(os.statvfs('/').f_bavail*os.statvfs('/').f_frsize)
""".replace('PATH',repr(remote)).replace('META',repr(meta))
 free=int(subprocess.check_output(S+['/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],text=True));meta.update(removed=True,free_bytes=free);receipt.write_text(json.dumps(meta,indent=2));print('migrated',name,'free GiB',free/2**30,flush=True)
 if free/2**30>=55:break
