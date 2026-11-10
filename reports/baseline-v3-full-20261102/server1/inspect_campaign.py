"""Real CPU-only inspection; does not pretend to repair pending items."""
import json,pathlib,subprocess,time,os
P=pathlib.Path(__file__).resolve().parent
s=json.loads((P/'STATE.json').read_text()) if (P/'STATE.json').exists() else {}
p=json.loads((P/'PROGRESS.json').read_text()) if (P/'PROGRESS.json').exists() else {}
q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495','npu-guard status --json; df -B1 /opt'],capture_output=True,text=True,timeout=30)
identity=False
try:identity=pathlib.Path('/proc/'+str(s['pid'])+'/stat').read_text().split()[21]==s['start_ticks']
except (FileNotFoundError,KeyError):pass
r={'checked_at':time.time(),'controller_identity_matches':identity,'state':s,'accepted':sum(x['status']=='accepted' for x in p.values()),'pending':[dict(id=k,**v) for k,v in p.items() if v['status']=='pending'],'remote_returncode':q.returncode,'remote_status':q.stdout,'remote_error':q.stderr,'action':'read PID/start_ticks, live host guard and disk; pending diagnosis remains agent responsibility'}
(P/('INSPECTION_'+str(time.time_ns())+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
