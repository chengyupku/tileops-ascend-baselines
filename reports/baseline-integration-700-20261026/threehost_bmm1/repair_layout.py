from pathlib import Path
import subprocess,shlex
P=Path(__file__).resolve().parent;R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_bmm1';SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
s=f'''from pathlib import Path
import shutil,json,hashlib
p=Path({R!r});a=p/'attempt2_metadata_root';a.mkdir()
for name in ['guard.log','guard.exit','GUARD_FINAL.json','FROZEN.json','capture.py','run.sh']:shutil.copy2(p/name,a/name)
shutil.copytree(p/'square-4k-bfloat16',a/'case')
for name in ['TileOPs','tools']:shutil.copytree(p/'frozen_workspace'/name,p/name)
for name in ['capture.py','run.sh']:
 x=p/name;x.write_text(x.read_text().replace(str(p/'frozen_workspace'),str(p)))
pins={{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for d in ['TileOPs','tools','private_provider'] for x in (p/d).rglob('*') if x.is_file()}}
for name in ['adapter.py','capture.py','native_observation.py','resource_gate.py','binding.py','CASES.json','run.sh']:pins[str(p/name)]=hashlib.sha256((p/name).read_bytes()).hexdigest()
(p/'FROZEN.json').write_text(json.dumps({{'sha256':pins}},indent=2))
print('private metadata root repaired')
'''
subprocess.run([SSH,'openlibing-DevEnv-661495','python3 -c '+shlex.quote(s)],check=True)
for name in ['capture.py','run.sh']:
 x=P/name;x.write_text(x.read_text().replace(R+'/frozen_workspace',R))
