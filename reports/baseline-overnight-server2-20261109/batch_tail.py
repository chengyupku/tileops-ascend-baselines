import pathlib,subprocess,json,time
P=pathlib.Path(__file__).resolve().parent
probe=P/'runs/600/topp-v2-probe'
while True:
 try:g=json.loads((probe/'guard.json').read_text())
 except (OSError,ValueError):g={}
 if g.get('phase')=='finished' or (probe/'device_failure.json').exists():break
 time.sleep(5)
patterns=['*-gqa.json']
if g.get('performance_valid') and g.get('workload_exit_code')==0 and (probe/'probe.json').exists():patterns.insert(0,'*-topp-v2.json')
result=[]
for pattern in patterns:
 for task in sorted((P/'tasks').glob(pattern)):
  t=json.loads(task.read_text());root=P/'runs'/str(t['ordinal']);prior=sorted(root.glob('tail-v3-a*')) if root.exists() else []
  valid=False
  for d in prior:
   try:h=json.loads((d/'guard.json').read_text())
   except (OSError,ValueError):h={}
   if h.get('phase')=='finished' and h.get('performance_valid') and h.get('workload_exit_code')==0 and (d/'measurement.json').exists():valid=True
  if valid:continue
  label='tail-v3-a'+str(len(prior)+1)
  r=subprocess.run(['python3',str(P/'run_one_resume.py'),task.name,label,'device'])
  result.append({'task':task.name,'label':label,'exit':r.returncode});(P/'TAIL_BATCH.json').write_text(json.dumps(result,indent=2))
  if r.returncode in (75,78):raise SystemExit(r.returncode)
print('tail torch batch completed',flush=True)
