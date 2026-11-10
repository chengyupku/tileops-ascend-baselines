from pathlib import Path
import subprocess,io,tarfile,json
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host='openlibing-DevEnv-055409';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_isnan_unpack1'
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for n in ['SCOPE.json','element_adapter.py','run_isnan.py','preflight.py']:t.add(P/n,arcname=n)
r=subprocess.run([ssh,host,'mkdir -p '+remote+' && tar xzf - -C '+remote],input=b.getvalue(),capture_output=True);assert r.returncode==0,r.stderr
r=subprocess.run([ssh,host,'python3 '+remote+'/preflight.py'],capture_output=True);(P/'preflight.stdout').write_bytes(r.stdout);(P/'preflight.stderr').write_bytes(r.stderr);print(r.stdout.decode()[-2000:]);assert r.returncode==0,r.stderr
