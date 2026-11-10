import subprocess,sys
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([SSH,'openlibing-DevEnv-661495',sys.argv[1]],input=sys.stdin.read(),text=True)
raise SystemExit(r.returncode)
