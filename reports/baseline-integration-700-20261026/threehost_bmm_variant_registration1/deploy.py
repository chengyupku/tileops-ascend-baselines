"""CPU-only prepare same-name remote directory; never launch numerical evaluation."""
import io,tarfile,subprocess,json,shlex,datetime
from pathlib import Path
O=Path(__file__).resolve().parent
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/'+O.name
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as tar:
 for p in sorted(O.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname=str(p.relative_to(O)))
r=subprocess.run([SSH,'openlibing-DevEnv-661495','mkdir -p '+shlex.quote(R)+' && tar xzf - -C '+shlex.quote(R)],input=stream.getvalue(),capture_output=True,timeout=60)
assert r.returncode==0,r.stderr
r=subprocess.run([SSH,'openlibing-DevEnv-661495','PYTHONDONTWRITEBYTECODE=1 /opt/tileops-ci/runtime/bin/python -I -S -B '+shlex.quote(R+'/cpu_checks.py')],capture_output=True,text=True,timeout=60)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_directory=R,rc=r.returncode,stdout=r.stdout,stderr=r.stderr,NPU_runs=0)
(O/'REMOTE_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
