import subprocess,io,tarfile,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent;ssh=['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409'];remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/register_server2/repair01'
data=subprocess.run(ssh+['tar -cf - -C '+remote+' .'],stdout=subprocess.PIPE,check=True).stdout
with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as t:
 for member in t.getmembers():
  if not (D/member.name).resolve().is_relative_to(D):raise ValueError('bad tar path')
 t.extractall(D,filter='data')
changes=json.loads((D/'CHANGED_FILES.json').read_text())
for row in changes:
 p=D/'provider'/Path(row['remote_path']).name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['after_sha256']
print('mirrored remote report bytes',len(data),'provider hash matches',len(changes))
