from pathlib import Path
import subprocess,sys,shlex
P=Path(__file__).resolve().parent;W=P.parents[2];ssh=str(W/'.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh');host={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[sys.argv[1]];remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_pto_original23'
if sys.argv[2]=='send':
 subprocess.run([ssh,host,'mkdir -p '+remote],check=True)
 for name in sys.argv[3:]:
  with (P/name).open('rb') as f:subprocess.run([ssh,host,'cat > '+remote+'/'+name],stdin=f,check=True)
else:
 r=subprocess.run([ssh,host,sys.argv[2]],capture_output=True,text=True);print(r.stdout);print(r.stderr,file=sys.stderr);sys.exit(r.returncode)
