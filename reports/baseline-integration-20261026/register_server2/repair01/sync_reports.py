import io,tarfile,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409'];remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/register_server2/repair01'
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w') as t:
 for f in D.iterdir():
  if f.is_file() and (f.suffix=='.py' or f.name=='DEPLOYMENT.md'):t.add(f,arcname=f.name)
subprocess.run(ssh+['tar -xmf - -C '+remote],input=b.getvalue(),check=True)
print('report scripts and deployment doc copied')
