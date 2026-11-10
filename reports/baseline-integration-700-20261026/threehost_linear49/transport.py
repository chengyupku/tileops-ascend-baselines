from pathlib import Path
import subprocess,tarfile,io,sys
P=Path(__file__).resolve().parent;remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_linear49';h=sys.argv[1];SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';host={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[h]
if sys.argv[2]=='push':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w:gz') as t:
  for n in ['snapshot','package','CONTRACTS.json','CHUNK_CASES.json','SCOPE.json','SNAPSHOT_HASHES.json','remote_run.py']:t.add(P/n,arcname=n)
 r=subprocess.run([SSH,host,f'mkdir -p {remote} && tar -xzf - -C {remote}'],input=b.getvalue());sys.exit(r.returncode)
else:
 name=sys.argv[2];r=subprocess.run([SSH,host,f'tar -czf - -C {remote} {name}'],capture_output=True,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:gz') as t:t.extractall(P/h,filter='data')
