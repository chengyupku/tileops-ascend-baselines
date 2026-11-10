from pathlib import Path
import subprocess,tarfile,io,json,shlex
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1';SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
script=f'''from pathlib import Path
import tarfile,sys
p=Path({R!r})
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for x in p.rglob('*'):
  rel=x.relative_to(p)
  if x.is_file() and not any(d in rel.parts for d in ['TileOPs','tools','frozen_workspace','private_provider']) and x.suffix in ['.json','.log','.exit','.py','.sh']:
   tar.add(x,arcname=str(rel))
'''
r=subprocess.run([SSH,'openlibing-DevEnv-661495','python3 -c '+shlex.quote(script)],stdout=subprocess.PIPE,check=True)
(P/'remote_receipts.tar.gz').write_bytes(r.stdout)
Q=P/'server1';Q.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as tar:tar.extractall(Q,filter='data')
print('collected',len(r.stdout))
