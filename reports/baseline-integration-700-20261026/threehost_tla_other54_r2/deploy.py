import pathlib,subprocess,json,shlex
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2];ssh=W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_tla_other54_r2'
for name,src in [('run_case.py',P/'run_case.py'),('SCOPE.json',P/'SCOPE.json'),('tilelang_adapter.py',P.parent/'tla_sdk_recovery36/integration700_tla_sdk_recovery36_impl.py')]:
 r=subprocess.run([str(ssh),'openlibing-DevEnv-661495','cat > '+shlex.quote(remote+'/'+name)],input=src.read_bytes());assert r.returncode==0
