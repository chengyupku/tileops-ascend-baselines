import pathlib,subprocess,tarfile
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tla_other54_r2'
r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','python3 -'],input=(P/'package_cpu.py').read_bytes(),capture_output=True);(P/'package_cpu.log').write_bytes(r.stdout+r.stderr);assert r.returncode==0
archive=P/'server1_evidence.tar.gz'
with archive.open('wb') as f:r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','tar -czf - -C '+remote+' .'],stdout=f,stderr=subprocess.PIPE);assert r.returncode==0,r.stderr
out=P/'server1_evidence';out.mkdir(exist_ok=True)
with tarfile.open(archive) as t:t.extractall(out,filter='data')
print('evidence bytes',archive.stat().st_size)
