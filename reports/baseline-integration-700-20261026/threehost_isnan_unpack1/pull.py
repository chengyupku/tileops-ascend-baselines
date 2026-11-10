from pathlib import Path
import subprocess,tarfile,io,json
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_isnan_unpack1'
r=subprocess.run([ssh,'openlibing-DevEnv-055409','tar czf - -C '+remote+' runs preflight.json headers_manifest.json headers_snapshot.tar.gz health_initial.json'],capture_output=True);assert r.returncode==0,r.stderr
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:t.extractall(P,filter='data')
r=subprocess.run([ssh,'openlibing-DevEnv-055409','npu-guard status --json'],capture_output=True);(P/'postflight.json').write_bytes(r.stdout)
print('Evidence pulled, bytes',len(r.stdout))
