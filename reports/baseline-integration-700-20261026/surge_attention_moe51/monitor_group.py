import pathlib,json,subprocess,datetime,time,collections
P=pathlib.Path(__file__).resolve().parent
now=time.time();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default
def write(name,x):
 p=P/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def identity(pid,stamp):
 try:
  s=pathlib.Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return boot+':'+s[19]==stamp and s[0]!='Z'
 except OSError:return False
rows=[];keys=[]
for j in read(P/'CHILD_JOBS.json',{}).get('jobs',[]):
 n=j['native'];r=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True)
 try:s=json.loads(r.stdout)
 except ValueError:s={'status':'status_error','error':r.stderr[:300]}
 d=pathlib.Path(j['directory']);scope=read(d/'SCOPE.json',[]);keys.extend(tuple(v['key']) for v in scope)
 artifacts=[{'path':str(f.relative_to(P)),'mtime':f.stat().st_mtime,'bytes':f.stat().st_size} for f in d.iterdir() if f.is_file() and f.name not in ('SCOPE.json','TASK.txt','SPAWN.json','COORDINATOR_NOTE.txt')]
 hb=read(d/'HEARTBEAT.json',{})
 rows.append({'name':d.name,'id':n['id'],'status':s.get('status'),'exit_code':s.get('exit_code'),'notification':s.get('notification'),'child_pid':s.get('child_pid'),'child_stamp':s.get('child_stamp'),'identity_alive':identity(s.get('child_pid'),s.get('child_stamp')),'supervisor_alive':identity(s.get('supervisor_pid'),s.get('supervisor_stamp')),'scope_count':len(scope),'heartbeat':hb,'artifacts':artifacts,'sampled_at':utc})
accepted=read(P.parent/'ACCEPTED.json',{});ar=accepted.get('rows',[])
a={'utc':utc,'active':sum(x['status'] in ('starting','running') for x in rows),'alive':sum(x['identity_alive'] for x in rows),'CLI completed':sum(x['status']=='completed' for x in rows),'fullchecker passed':None,'pending registry':0,'root accepted':len(ar),'waiting_host':None,'stalled':[x['name'] for x in rows if x['status'] in ('starting','running') and not x['identity_alive']],'replacement_actions':[],'accepted_added':0,'rolling60_accepted_added':None,'model_queue_seconds':None,'timing':{'npu_queue_seconds':None,'cpu_preparation_seconds':None,'fullchecker_seconds':None,'review_seconds':None},'scope_keys':len(keys),'unique_keys':len(set(keys)),'ownership_disjoint':len(keys)==len(set(keys)),'workers':rows}
result=read(P/'RESULT.json',{})
if result.get('scope_total')==51:
 a['fullchecker passed']=result.get('new_fullchecker_passed')
 a['remaining_scope']=51
 a['waiting_host']=0
 a['replacement_actions']=[{'action':'no_replacement','reason':result.get('replacement_decision')}]
 a['no_device_submission_this_round']=True
 a['terminal_blocked_scope_count']=51
samples=[]
if (P/'PROCESS_SAMPLES.jsonl').exists():
 for line in (P/'PROCESS_SAMPLES.jsonl').read_text().splitlines():
  try:samples.append(json.loads(line))
  except ValueError:pass
active_samples=sum(v['active'] for v in samples)+a['active'];alive_samples=sum(v['alive'] for v in samples)+a['alive']
a['process_identity_sample_survival_rate']=alive_samples/active_samples if active_samples else None
a['terminal_failure_rate']=sum(x['status'] in ('failed','lost','timeout','cancelled') for x in rows)/len(rows) if rows else None
series=samples+[a];occupied=duration=0
for left,right in zip(series,series[1:]):
 dt=datetime.datetime.fromisoformat(right['utc']).timestamp()-datetime.datetime.fromisoformat(left['utc']).timestamp();duration+=dt;occupied+=dt*left['active']
a['time_weighted_slot_occupancy']=occupied/(duration*7) if duration else None
a['rolling60_accepted_added']=None
a['observed_root_accepted_delta']=a['root accepted']-samples[0]['root accepted'] if samples else 0
a['rolling60_note']='Observation shorter than 60 minutes; historical acceptance timestamps not audited.'
write('GROUP_HEARTBEAT.json',a);write('GROUP_STATUS.json',a)
with (P/'PROCESS_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(a,ensure_ascii=False)+'\n')
if not (P/'PUBLISH_QUEUE.json').exists():write('PUBLISH_QUEUE.json',{'utc':utc,'entries':[],'accepted_added':0,'reason':'No reviewed fullchecker success yet; parent is sole shared registry writer.'})
print(json.dumps({k:v for k,v in a.items() if k!='workers'},ensure_ascii=False))
for x in rows:print(x['name'],x['status'],'alive='+str(x['identity_alive']),'files='+','.join(y['path'].split('/')[-1] for y in x['artifacts']),str(x['heartbeat'])[:500])
