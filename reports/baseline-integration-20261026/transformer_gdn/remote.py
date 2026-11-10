import subprocess,sys
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
p=subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -'],input=sys.stdin.read(),text=True)
sys.exit(p.returncode)
