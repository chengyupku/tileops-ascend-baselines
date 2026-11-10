"""Serial guarded formal batches. Start only after explicit pilot gate file exists."""
import pathlib,json,subprocess,sys,collections,datetime
P=pathlib.Path(__file__).resolve().parent
G=P/'FORMAL_GATE.json'
assert G.exists(),'Pilot review must authorize exact op/variant combinations first'
gate=json.loads(G.read_text());assert gate['ready']
allowed={tuple(x) for x in gate['allowed_op_variants']}
tasks=json.loads((P/'TASKS.json').read_text());jobs=json.loads((P/'FORMAL_JOBS.pending.json').read_text())
failed=collections.Counter();records=[]
priority=list(range(20,40))+list(range(40,48))+list(range(5,12))+list(range(0,5))+list(range(12,20))+list(range(48,71))
assert sorted(priority)==list(range(71))
for chunk in range(0,len(tasks),2):
 if (P/'FORMAL_STOP').exists():break
 batch=[]
 for j in jobs:
  if j['index'] not in priority[chunk:chunk+2]:continue
  key=(tasks[j['index']]['op'],j['variant'])
  if key not in allowed or failed[key]>=2:
   records.append({'job':j,'status':'deferred_unmeasured','reason':'pilot gate absent' if key not in allowed else 'two same-family failures'});continue
  assert not (P/f"remote/runs/{j['index']}/{j['variant']}/{j['attempt']}/result.json").exists(),'Immutable attempt already exists'
  batch.append(j)
 if not batch:continue
 name=f'batch{10+chunk//2:02d}';path=P/(name+'.json');path.write_text(json.dumps(batch,indent=2))
 rc=subprocess.run([sys.executable,str(P/'run_batch_v10.py'),str(path)]).returncode
 subprocess.run([sys.executable,str(P/'audit_results.py')],check=True)
 guard=P/'remote'/(name+'-guard.json')
 g=json.loads(guard.read_text()) if guard.exists() else {}
 if rc or not g.get('performance_valid') or g.get('exit_code')!=0:
  records.append({'batch':name,'status':'stopped_guard_failure','exit':rc});break
 for j in batch:
  f=P/f"remote/runs/{j['index']}/{j['variant']}/{j['attempt']}/result.json"
  m=json.loads(f.read_text()) if f.exists() else {'status':'missing'}
  records.append({'job':j,'status':m['status'],'error':m.get('error')})
  if m['status'] in ('failed','missing'):failed[(tasks[j['index']]['op'],j['variant'])]+=1
  if j['variant']=='original' and m['status'] in ('failed','missing'):
   (P/'FORMAL_STOP').write_text('Original control failed: inspect before continuing\n')
 (P/'CAMPAIGN_PROGRESS.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2))
(P/'CAMPAIGN_PROGRESS.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'controller_finished':True},indent=2))
