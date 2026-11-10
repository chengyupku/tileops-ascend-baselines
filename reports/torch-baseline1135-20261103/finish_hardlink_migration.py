import json,pathlib,subprocess,shlex,time
P=pathlib.Path(__file__).resolve().parent;m=json.loads((P/'migration/reference_0_input1.bin.json').read_text());assert m['verified'] and m['removed']
code='''import pathlib,json,hashlib,os
m=META;p=pathlib.Path(m['remote']).parent
names=['reference_0_post_input1.bin','candidate_0_input1.bin','candidate_0_post_input1.bin'];fs=[p/n for n in names];st=[f.stat() for f in fs]
assert len({(s.st_dev,s.st_ino) for s in st})==1
assert all(s.st_nlink==3 and s.st_size==m['size'] for s in st)
identity=(st[0].st_dev,st[0].st_ino)
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  for fd in (proc/'fd').iterdir():
   try:
    x=fd.stat();assert (x.st_dev,x.st_ino)!=identity,'capture still open'
   except FileNotFoundError:pass
 except PermissionError:raise RuntimeError('cannot verify handles')
assert hashlib.file_digest(fs[0].open('rb'),'sha256').hexdigest()==m['sha256']
for f in fs:
 x=f.stat();assert (x.st_dev,x.st_ino)==identity;f.unlink()
print(json.dumps({'removed':[str(f) for f in fs],'same_sha256':m['sha256'],'archive':m['archive'],'free_bytes':os.statvfs('/').f_bavail*os.statvfs('/').f_frsize}))
'''.replace('META',repr(m))
r=subprocess.check_output(['ssh','-T','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','/opt/tileops-ci/runtime/bin/python -I -S -c '+shlex.quote(code)],text=True);d=json.loads(r);d['at']=time.time();(P/'migration/HARDLINK_MIGRATION.json').write_text(json.dumps(d,indent=2));print(r)
