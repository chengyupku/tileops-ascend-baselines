import subprocess,sys,pathlib
P=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([SSH,'openlibing-DevEnv-055409',sys.argv[1]],input=sys.stdin.buffer.read(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(P/sys.argv[2]).write_bytes(r.stdout); print(r.stdout.decode(errors='replace'));sys.exit(r.returncode)
