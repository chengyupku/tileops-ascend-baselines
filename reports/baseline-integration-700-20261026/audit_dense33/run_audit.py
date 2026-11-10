import pathlib,subprocess,json,os
D=pathlib.Path(__file__).resolve().parent
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
code=(D/'remote_audit.py').read_text().replace('SCOPE_PLACEHOLDER',repr(json.loads((D/'SCOPE.json').read_text())))
cmd=[ssh,'openlibing-DevEnv-661495','PYTHONDONTWRITEBYTECODE=1 python3 -S -']
p=subprocess.run(cmd,input=code,text=True,capture_output=True,timeout=180)
(D/'SSH_COMMAND.json').write_text(json.dumps(dict(command=cmd,exit=p.returncode,stderr=p.stderr),indent=2))
(D/'REMOTE_AUDIT.json').write_text(p.stdout)
print('ssh exit',p.returncode,'bytes',len(p.stdout),'stderr',p.stderr[-2500:])
