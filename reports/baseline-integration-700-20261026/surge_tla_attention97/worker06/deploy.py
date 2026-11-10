from pathlib import Path
import tarfile,subprocess,sys,shlex,json
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker06'
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
with tarfile.open(P/'deployment.tar','w') as t:
 for n in ['run_case.py','adapter.py','capture.py','private_tla','frozen','originals','SCOPE.json']:
  t.add(P/n,arcname=n,filter=lambda x:None if '__pycache__' in x.name else x)
for host in sys.argv[1:]:
 with (P/'deployment.tar').open('rb') as f:r=subprocess.run([SSH,host,'mkdir -p '+shlex.quote(R)+' && tar -xf - -C '+shlex.quote(R)],stdin=f,capture_output=True,text=False)
 (P/(host+'_deployment.json')).write_text(json.dumps(dict(exit=r.returncode,stderr=r.stderr.decode(),path=R)));print(host,r.returncode)
