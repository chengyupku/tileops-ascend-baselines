from pathlib import Path
import subprocess,tarfile
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tilelang_bmm_native3'
s=(P/'package_cpu.py').read_text().replace("P/'DEPENDENCY_ARTIFACT_HASHES.json'","P/'DEPENDENCY_ARTIFACT_HASHES_FINAL.json'")
r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','python3 -'],input=s.encode(),capture_output=True);(P/'package_final.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0
archive=P/'closure.tar.gz'
with archive.open('xb') as f:
 r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','tar -czf - -C '+remote+' frozen tilelang_adapter.py SNAPSHOT.json CPU_PREFLIGHT.json DEPENDENCY_ARTIFACT_HASHES_FINAL.json source_freeze cache run_case.py SCOPE.json'],stdout=f,stderr=subprocess.PIPE);assert r.returncode==0,r.stderr
with tarfile.open(archive) as t:t.extractall(P/'closure',filter='data')
print('closure bytes',archive.stat().st_size)
