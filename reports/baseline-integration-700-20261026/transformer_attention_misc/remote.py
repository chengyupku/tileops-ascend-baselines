import subprocess,sys,pathlib
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
r=subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -'],input=sys.stdin.buffer.read(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
sys.stdout.buffer.write(r.stdout)
