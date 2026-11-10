"""Foreground continuation; no device work except bounded run_one guard."""
import pathlib,subprocess,json,time
P=pathlib.Path(__file__).resolve().parent
while True:
 f=P/'W4_BATCH.json';rows=json.loads(f.read_text()) if f.exists() else []
 if rows and (len(rows)==15 or rows[-1]['exit']):break
 time.sleep(5)
result=[]
def attempt(task,label,mode):
 t=json.loads(task.read_text());out=P/'runs'/str(t['ordinal'])/label
 if out.exists():
  g=json.loads((out/'guard.json').read_text()) if (out/'guard.json').exists() else {}
  return 0 if g.get('phase')=='finished' and g.get('performance_valid') and g.get('workload_exit_code')==0 and (out/('probe.json' if mode=='probe' else 'measurement.json')).exists() else 99
 r=subprocess.run(['python3',str(P/'run_one.py'),task.name,label,mode])
 result.append({'task':task.name,'label':label,'mode':mode,'exit':r.returncode})
 (P/'FOLLOW_BATCH.json').write_text(json.dumps(result,indent=2));return r.returncode
for pattern,label,probe in [('*-kda.json','kda-v3',None),('*-cumprod.json','cumprod-v3',None),('*-sampling.json','sampling-v3','sampling-probe'),('*-nsa.json','nsa-v3','nsa-probe')]:
 tasks=sorted((P/'tasks').glob(pattern))
 if probe and attempt(tasks[0],probe,'probe'):continue
 for task in tasks:
  rc=attempt(task,label,'device')
  if rc in (75,78):raise SystemExit(rc)
  if rc:break
print('follow batch complete',flush=True)
