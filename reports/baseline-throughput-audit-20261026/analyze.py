import json,pathlib,datetime,collections,re,ast,statistics,csv
P=pathlib.Path(__file__).resolve().parent
def epoch(x):
 if isinstance(x,(int,float)):return x
 try:return datetime.datetime.fromisoformat(x.replace('Z','+00:00')).timestamp()
 except Exception:return None
def union(intervals):
 total=0;end=-1e30
 for a,b in sorted(intervals):
  total+=max(0,b-max(a,end));end=max(end,b)
 return total
allworkers=[];guardrows=[];fleet={}
for host in ('local','server1','server2'):
 x=json.loads((P/(host+'.json')).read_text());now=x['host_now'];guards={}
 def addguard(v):
  if not isinstance(v,dict) or not v.get('owner') or not v.get('requested_at'):return
  k=(v['owner'],v['requested_at']);g=guards.setdefault(k,{})
  for f,val in v.items():
   if val is not None:g[f]=val
 def textguards(txt):
  for line in txt.splitlines():
   if "'requested_at':" not in line:continue
   a=line.find('{');b=line.rfind('}')
   if a>=0 and b>a:
    try:addguard(ast.literal_eval(line[a:b+1]))
    except Exception:pass
 addguard(x.get('guard_lock_metadata'))
 for q in x['queue']:addguard(q.get('data'))
 for j in x['jobs']:
  for a in j.get('artifacts',[]):addguard(a.get('data'));textguards(a.get('text',''))
  for c in j.get('commands',[]):textguards(c.get('output_tail',''))
 for g in guards.values():
  req=epoch(g['requested_at']);acq=epoch(g.get('acquired_at'))
  g.update(host=host,request_epoch=req,acquired_epoch=acq,wait_s=acq-req if acq else None,observed_pending_s=max(0,now-req) if not acq else None)
  guardrows.append(g)
 recent=[j for j in x['jobs'] if j['role']=='worker']
 last_hour=[j for j in recent if j['created_at']<=now and (j.get('finished_at') or now)>=now-3600]
 fleet[host]={'host_now':now,'clock_offset_s':x['clock_offset_s'],'recent6h_status_counts':dict(collections.Counter(j['status'] for j in recent)),
  'last_hour_overlapping_job_statuses':dict(collections.Counter(j['status'] for j in last_hour)),
  'live_status_claims':sum(j['status'] in ('running','starting','cancelling') for j in recent),'actual_live':sum(j['alive'] for j in recent),
  'last_hour_worker_seconds':sum(max(0,min(now,j.get('finished_at') or now)-max(now-3600,j['created_at'])) for j in recent)}
 for j in x['jobs']:
  if not j['selected_active']:continue
  name=j['write'][0].split('/')[-1];duration=min(now,j.get('finished_at') or now)-j['created_at']
  gs=[g for g in guards.values() if g['owner'].endswith('/'+name)]
  valid=[];checks=[]
  for a in j.get('artifacts',[]):
   d=a.get('data',{});v=d.get('verification',{}) if isinstance(d,dict) else {}
   if isinstance(d,dict) and d.get('case_id') and d.get('op') and d.get('library'):
    checks.append((a,d))
    if d.get('status')=='verified' and d.get('inputs_preserved') and v.get('total_outputs',0)>0 and v.get('checked_outputs')==v.get('total_outputs'):valid.append((a,d))
  waits=[g['wait_s'] for g in gs if g['wait_s'] is not None]
  pending=[g['observed_pending_s'] for g in gs if g['acquired_epoch'] is None]
  calls=j.get('calls',[])
  intervals=[(c['start'],c['end']) for c in calls if c.get('end')]
  # Time in a tool call is observable wall time, not token generation/model utilization.
  sleep=union([(c['start'],c['end']) for c in calls if c.get('end') and c['name']=='sleep'])
  guard_artifacts=[a['data'] for a in j.get('artifacts',[]) if isinstance(a.get('data'),dict) and 'guard' in a['path'].lower()]
  starts=[epoch(g.get('started_at',g.get('start'))) for g in guard_artifacts]
  starts=[v for v in starts if v]+[g['request_epoch'] for g in gs]
  row={'host':host,'id':j['id'],'task':name,'status':j['status'],'alive':j['alive'],'created_at':j['created_at'],'elapsed_s':duration,
   'command_count':j['command_count'],'nonzero_commands':j['command_nonzero'],'tool_calls':len(calls),'tool_wall_union_s':union(intervals),'sleep_wall_s':sleep,
   'first_guard_delay_s':min(starts)-j['created_at'] if starts else None,'full_verified_receipts':len(valid),'checker_statuses':dict(collections.Counter(d.get('status') for a,d in checks)),
   'checker_wall_sum_s':sum(d.get('wall_seconds') or 0 for a,d in checks),
   'queue_observations':len(gs),'acquired_observations':len(waits),'queue_wait_sum_s':sum(waits),'queue_wait_mean_s':statistics.mean(waits) if waits else None,'queue_wait_max_s':max(waits) if waits else None,'pending_age_s':max(pending) if pending else None,
   'latest_messages':j['messages'][-2:],'source':str(P/(host+'.json'))}
  allworkers.append(row)
summary={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'agents':allworkers,'hosts':fleet,'guard_records':guardrows}
(P/'METRICS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (P/'AGENTS.csv').open('w') as f:
 fields=[k for k in allworkers[0] if k not in ('latest_messages','checker_statuses')];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(allworkers)
for h,v in fleet.items():print(h,json.dumps(v,ensure_ascii=False))
for a in allworkers:print(a['host'],a['id'],a['task'],'min',round(a['elapsed_s']/60,2),'commands',a['command_count'],'pass',a['full_verified_receipts'],'checksec',round(a['checker_wall_sum_s'],1),'pre',round((a['first_guard_delay_s'] or 0)/60,1),'wait_sum',round(a['queue_wait_sum_s']/60,1),'pending',round((a['pending_age_s'] or 0)/60,1),'sleep',round(a['sleep_wall_s']/60,1))
