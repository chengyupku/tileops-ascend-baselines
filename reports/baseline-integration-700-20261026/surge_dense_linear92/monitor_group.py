import json,pathlib,subprocess,datetime,time,collections
P=pathlib.Path(__file__).resolve().parent
now=time.time();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return default
def stamp(pid):
 try:
  a=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return None if a[0]=='Z' else pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+a[19]
 except OSError:return None
def write(name,obj):
 t=P/(name+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(P/name)
rows=[];seen=set();owned=set();dups=[]
for j in read(P/'CHILD_JOBS.json',{'jobs':[]})['jobs']:
 n=j['native'];r=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True)
 try:s=json.loads(r.stdout)
 except ValueError:s={'status':'status_unavailable','error':r.stderr[-500:]}
 q=pathlib.Path(j['directory']);scope=read(q/'SCOPE.json',[])
 for x in scope:
  key=tuple(x.get('key') or [x.get(k) for k in ['baseline','operator','case_id']])
  if j.get('active_assignment',True):
   if key in owned:dups.append(key)
   owned.add(key)
  seen.add(key)
 hbf=max((f for f in [q/'HEARTBEAT.json',q/'HEARTBEAT.local.json'] if f.exists()),key=lambda f:f.stat().st_mtime,default=None)
 hb=read(hbf,{}) if hbf else {}
 artifacts=[f for f in q.iterdir() if f.name not in ['SCOPE.json','TASK.txt','SPAWN.json']]
 latest=max((f.stat().st_mtime for f in artifacts),default=0)
 rows.append({'name':j['name'],'job_id':n['id'],'status':s.get('status'),'exit_code':s.get('exit_code'),'notification':s.get('notification'),'child_pid':s.get('child_pid'),'child_stamp':s.get('child_stamp'),'child_alive_identity':bool(s.get('child_stamp') and stamp(s.get('child_pid'))==s['child_stamp']),'supervisor_alive_identity':bool(s.get('supervisor_stamp') and stamp(s.get('supervisor_pid'))==s['supervisor_stamp']),'scope_count':len(scope),'heartbeat':hb,'heartbeat_age_s':max(0,now-hbf.stat().st_mtime) if hbf else None,'last_artifact_age_s':max(0,now-latest) if latest else None,'artifacts':[f.name for f in artifacts]})
accepted=read(P.parent/'ACCEPTED.json',{}).get('rows',[])
accepted_keys={tuple(r.get('key') or [r.get(k) for k in ['baseline','operator','case_id']]) for r in accepted}
pq=read(P/'PUBLISH_QUEUE.json',{'entries':[]})
rolling=[]
for r in accepted:
 try:
  if now-3600 <= datetime.datetime.fromisoformat(r.get('at','')).timestamp() <= now:rolling.append(r)
 except (ValueError,TypeError):pass
previous=[]
if (P/'IDENTITY_SAMPLES.jsonl').exists():
 for l in (P/'IDENTITY_SAMPLES.jsonl').read_text().splitlines():
  try:previous.append(json.loads(l))
  except ValueError:pass
active_samples=sum(x.get('active',0) for x in previous)+sum(r['status'] in ['running','starting'] for r in rows)
alive_samples=sum(x.get('alive',0) for x in previous)+sum(r['child_alive_identity'] for r in rows)
weighted=sum((b['sample_at_epoch']-a['sample_at_epoch'])*a.get('active',0) for a,b in zip(previous,previous[1:]))
elapsed=(now-previous[0]['sample_at_epoch']) if previous else 0
if previous:weighted+=(now-previous[-1]['sample_at_epoch'])*previous[-1].get('active',0)
h={'root_accepted_rolling60m':len(rolling),'process_identity_sample_survival':alive_samples/active_samples if active_samples else None,'time_weighted_7slot_occupancy':weighted/(7*elapsed) if elapsed else None,'terminal_failure_rate':sum(r['status'] in ['failed','timeout','lost','cancelled'] for r in rows)/sum(r['status'] not in ['running','starting'] for r in rows) if any(r['status'] not in ['running','starting'] for r in rows) else None,'utc':utc,'sample_at_epoch':now,'active':sum(r['status'] in ['starting','running'] for r in rows),'alive':sum(r['child_alive_identity'] for r in rows),'CLI completed':sum(r['status']=='completed' for r in rows),'fullchecker passed':len(pq.get('entries',[])),'pending registry':sum(tuple(e['key']) not in accepted_keys for e in pq.get('entries',[])),'root accepted':len(accepted),'group_root_accepted':len(seen & accepted_keys),'accepted_added':0,'waiting_host':[r['name'] for r in rows if r['status']=='running' and any(x in (str(r['heartbeat'].get('phase',''))+' '+str(r['heartbeat'].get('last_output',''))).lower() for x in ['queue','waiting'])],'stalled':[r['name'] for r in rows if r['status']=='running' and r['last_artifact_age_s'] is not None and r['last_artifact_age_s']>300],'replacement_actions':read(P/'REPLACEMENT_ACTIONS.json',[]),'model_queue_s':None,'scope_count':len(seen),'duplicate_keys':dups,'workers':rows}
with (P/'IDENTITY_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(h,ensure_ascii=False)+'\n')
write('GROUP_HEARTBEAT.json',h)
write('HEARTBEAT.json',h)
status=P/'GROUP_STATUS.json'
if not status.exists() or now-status.stat().st_mtime>=290:write('GROUP_STATUS.json',h)
print(json.dumps({k:v for k,v in h.items() if k not in ['workers','replacement_actions']},ensure_ascii=False))
for r in [x for x in rows if x['status'] in ['running','starting','stopping']]:print(json.dumps({'worker':r['name'].split('/')[-1],'status':r['status'],'alive':r['child_alive_identity'],'heartbeat_age_s':r['heartbeat_age_s'],'artifact_age_s':r['last_artifact_age_s'],'phase':r['heartbeat'].get('phase'),'last_output':r['heartbeat'].get('last_output')},ensure_ascii=False))
