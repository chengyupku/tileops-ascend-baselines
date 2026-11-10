"""Formal eager/native first; compiler jobs use separate guarded reservations."""
import pathlib,json,subprocess,sys,datetime,collections
P=pathlib.Path(__file__).resolve().parent
T=json.loads((P/'TASKS.json').read_text());priority=list(range(20,40))+list(range(40,48))+list(range(5,12))+list(range(5))+list(range(12,20))+list(range(48,71))
records=[];counter=100;failures=collections.Counter()
def save():
 (P/'SEPARATED_V2_PROGRESS.json').write_text(json.dumps({'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2))
def submit(batch):
 global counter
 while (P/f'batch{counter}.json').exists():counter+=1
 name=f'batch{counter}';counter+=1;f=P/(name+'.json');f.write_text(json.dumps(batch,indent=2))
 subprocess.run([sys.executable,str(P/'run_batch_v16.py'),str(f)],check=True)
 subprocess.run([sys.executable,str(P/'audit_results.py')],check=True)
 subprocess.run([sys.executable,str(P/'build_comparison.py')],check=True)
 g=json.loads((P/'remote'/(name+'-guard.json')).read_text())
 records.append({'batch':name,'jobs':batch,'guard_valid':g.get('performance_valid'),'guard_exit':g.get('exit_code')});save()
 if not g.get('performance_valid') or g.get('exit_code')!=0:raise RuntimeError('Guard failure: review before continuing')
 for j in batch:
  f=P/f"remote/runs/{j['index']}/{j['variant']}/{j['attempt']}/result.json"
  m=json.loads(f.read_text()) if f.exists() else {'status':'missing'}
  if m['status'] in ('failed','missing'):
   failures[(T[j['index']]['op'],j['variant'])]+=1
   if j['variant']=='original':raise RuntimeError('Original control failed')
for compiler_phase in (False,True):
 batch=[]
 for i in priority:
  if (P/'FORMAL_STOP').exists():save();sys.exit(0)
  jobs=json.loads((P/'FORMAL_JOBS.pending.json').read_text());gate=json.loads((P/'FORMAL_GATE.json').read_text());allowed={tuple(x) for x in gate['allowed_op_variants']}
  audited=json.loads((P/'RESULTS.json').read_text())
  for original_job in jobs:
   if original_job['index']!=i or original_job['variant'].startswith('compile')!=compiler_phase:continue
   j=dict(original_job);v=j['variant'];key=(T[i]['op'],v)
   if key not in allowed or failures[key]>=2:
    records.append({'job':j,'status':'deferred_unmeasured','reason':'gate absent or family failure limit'});continue
   if any(x['index']==i and x['variant']==v and x['status']=='formal_raw_verified' for x in audited):continue
   params=json.loads((P/f'remote/runs/{i}/prepare.json').read_text())['params']
   if v=='compile' and T[i]['op']=='INT8DequantPerBlockFwdOp' and i!=20:
    records.append({'job':j,'status':'deferred_unmeasured','reason':'Repeated block broadcast fused kernel init wait, including synchronous compiler; keep native and compiled-native alternatives'});continue
   existing=P/f"remote/runs/{i}/{v}/{j['attempt']}/result.json"
   if existing.exists():
    m=json.loads(existing.read_text())
    if 'timing' not in m:records.append({'job':j,'status':'existing_failure_not_retried'});continue
    # Same measured implementation, previous enclosing guard invalid. Retain it;
    # require a new fully valid reservation before accepting performance.
    j['attempt']='formal2_guardretry'
    assert not (P/f"remote/runs/{i}/{v}/{j['attempt']}/result.json").exists()
   elif existing.parent.exists():
    # Preserve incomplete timeout evidence; process-local synchronous compile is new evidence.
    j['attempt']='formal2_synccompile'
    assert not (P/f"remote/runs/{i}/{v}/{j['attempt']}").exists()
   batch.append(j)
   if len(batch)>=(2 if compiler_phase else 6):submit(batch);batch=[]
 if batch:submit(batch)
save();print('SEPARATED_CAMPAIGN_DONE',flush=True)
