from pathlib import Path
import subprocess,tarfile,sys
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host=sys.argv[1] if len(sys.argv)>1 else 'server1';alias={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host];remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_pto_original23'
a=P/(host+'_evidence.tar.gz')
with a.open('wb') as f:subprocess.run([ssh,alias,'tar --exclude=./cache --exclude=./runtime/tilelang/jit/adapter/cython/.cycache -czf - -C '+remote+' .'],stdout=f,check=True)
d=P/(host+'_evidence');d.mkdir(exist_ok=True)
with tarfile.open(a) as t:t.extractall(d,filter='data')
print(a.stat().st_size)
