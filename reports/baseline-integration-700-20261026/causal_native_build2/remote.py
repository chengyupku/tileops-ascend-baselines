import subprocess,sys,pathlib
BASE=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/causal_native_build2')
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
p=subprocess.run([SSH,'openlibing-DevEnv-055409','bash -s'],input=sys.stdin.buffer.read(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
if len(sys.argv)>1:(BASE/sys.argv[1]).write_bytes(p.stdout)
sys.stdout.buffer.write(p.stdout);sys.exit(p.returncode)
