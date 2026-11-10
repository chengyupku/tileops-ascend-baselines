import subprocess,pathlib,sys,time,json
R=pathlib.Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def run(code):
 return subprocess.run([SSH,'openlibing-DevEnv-055409','python3 -'],input=code,text=True,capture_output=True)
if __name__=='__main__':
 p=run(sys.stdin.read()); print(p.stdout,end='');print(p.stderr,end='',file=sys.stderr);sys.exit(p.returncode)
