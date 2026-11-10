from remote import R,SSH
import subprocess,tarfile
code="""import tarfile,sys
from pathlib import Path
r=Path(%r)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for p in r.iterdir():
  if p.name in ('attempt01','attempt02','runtime_logs') or (p.is_file() and p.suffix in ('.json','.maps','.log','.exit','.py','.sh')):
   t.add(p,arcname=p.name)
""" % str(R)
archive=R/'remote_evidence.tar.gz'
with archive.open('wb') as f:
 x=subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -'],input=code.encode(),stdout=f,stderr=subprocess.PIPE)
assert x.returncode==0,x.stderr
with tarfile.open(archive) as t:t.extractall(R,filter='data')
print('evidence synchronized',archive.stat().st_size)
