from pathlib import Path
import subprocess,tarfile,io,sys
W=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker04r2'
ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409']
if sys.argv[1]=='push':
 b=io.BytesIO()
 with tarfile.open(fileobj=b,mode='w') as t:
  for n in ['package','ALL_CONTRACTS.json','remote_run.py','CONTRACT.json'] :t.add(W/n,arcname=n)
 subprocess.run(ssh+[f'mkdir -p {R} && tar -xf - -C {R}'],input=b.getvalue(),check=True)
elif sys.argv[1]=='get':
 a=sys.argv[2];p=subprocess.run(ssh+[f'tar -cf - -C {R} {a} HEARTBEAT.json'],capture_output=True,check=True)
 with tarfile.open(fileobj=io.BytesIO(p.stdout)) as t:t.extractall(W,filter='data')
