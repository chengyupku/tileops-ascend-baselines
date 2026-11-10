from pathlib import Path
import subprocess,tarfile,io,shlex,json
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh')
remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_isnan_adapter1'
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for n in ['frozen','runtime','private_tla','element_adapter.py','run_isnan.py','SCOPE.json']:t.add(P/n,arcname=n)
r=subprocess.run([ssh,'openlibing-DevEnv-055409','mkdir -p '+shlex.quote(remote)+' && tar xzf - -C '+shlex.quote(remote)],input=b.getvalue(),capture_output=True)
assert r.returncode==0,r.stderr
print('deployed bytes',len(b.getvalue()))
