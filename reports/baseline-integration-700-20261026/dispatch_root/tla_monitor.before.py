import pathlib,json,subprocess,time,datetime,collections
P=pathlib.Path(__file__).resolve().parent
now=time.time(); utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p,default=None):
 try:return json.loads(p.read_text())
 except:return default
def stamp(pid):
 try:return pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except:return None
rows=[];keys=[];results=[]
for r in read(P/'CHILD_JOBS.json',{}).get('jobs',[]):
 n=r['native']; z=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True)
 try:s=json.loads(z.stdout)
 except:s={'status':'status_error','error':z.stderr[-500:]}
 w=pathlib.Path(r['directory']);keys.extend(tuple(x['key']) for x in read(w/'SCOPE.json',[]))
 hb=read(w/'HEARTBEAT.json');result=read(w/'RESULT.json')
 rr=result if isinstance(result,list) else (result or {}).get('exactkeys',(result or {}).get('results',(result or {}).get('rows',[])))
 for item in rr:
  if isinstance(item,dict):results.append({'worker':w.name,'key':item.get('key'),'status':item.get('status'),'numerical_success':item.get('numerical_success'),'source':str(w/'RESULT.json')})
 rows.append({'worker':w.name,'job':n['id'],'status':s.get('status'),'exit_code':s.get('exit_code'),'child_pid':s.get('child_pid'),'child_stamp':s.get('child_stamp'),'identity_alive':bool(s.get('child_stamp') and stamp(s.get('child_pid'))==s.get('child_stamp')),'supervisor_alive':bool(s.get('supervisor_stamp') and stamp(s.get('supervisor_pid'))==s.get('supervisor_stamp')),'notification':s.get('notification'),'heartbeat_age_s':now-(w/'HEARTBEAT.json').stat().st_mtime if hb else None,'heartbeat':hb,'has_result':result is not None,'files':[f.name for f in w.iterdir() if f.name not in ['TASK.txt','SCOPE.json','SPAWN.json']]})
a=read(P.parent/'ACCEPTED.json',{}); ar=a.get('rows',[])
h={'utc':utc,'sample_epoch':now,'active':sum(r['status'] in ('starting','running') for r in rows),'alive':sum(r['identity_alive'] for r in rows),'CLI completed':sum(r['status']=='completed' for r in rows),'fullchecker passed':sum(r.get('numerical_success') is True for r in results),'pending registry':len(read(P/'PUBLISH_QUEUE.json',{}).get('entries',[])),'root accepted':len(ar),'root_accepted_updated_at':a.get('updated_at'),'waiting_host':None,'stalled':[],'replacement_actions':[json.loads(l).get('condition_changed') for l in (P/'REPLACEMENTS.jsonl').read_text().splitlines()] if (P/'REPLACEMENTS.jsonl').exists() else [],'scope_count':len(keys),'unique_scope_count':len(set(keys)),'accepted_added':0,'model_queue_seconds':None,'workers':rows}
h['group_scope_now_root_accepted']=[r['key'] for r in ar if tuple(r.get('key',[])) in set(keys) and r.get('accepted')]
h['rolling_60m_root_accepted_added']=sum(1 for r in ar if r.get('accepted') and r.get('at') and now-3600<=datetime.datetime.fromisoformat(r['at']).timestamp()<=now)
h['waiting_host']=[r['worker'] for r in rows if any(t in str(r.get('heartbeat') or {}).lower() for t in ['queue','reservation','admission','guard_wait','waiting_for_available'])]
h['waiting_host_qualification']='heartbeat-reported candidates; queue durations require original guard receipts'
previous=[json.loads(l) for l in (P/'PROCESS_SAMPLES.jsonl').read_text().splitlines()] if (P/'PROCESS_SAMPLES.jsonl').exists() else []
samples=previous+[h]
h['process_identity_sample_survival_rate']=sum(sum(r['identity_alive'] for r in x['workers'] if r['status'] in ['running','starting']) for x in samples)/max(1,sum(x['active'] for x in samples))
h['time_weighted_slot_occupancy']=sum((b['sample_epoch']-a['sample_epoch'])*a['alive'] for a,b in zip(samples,samples[1:]))/(7*(now-samples[0]['sample_epoch'])) if now>samples[0]['sample_epoch'] else None
replaced=[json.loads(l) for l in (P/'REPLACEMENTS.jsonl').read_text().splitlines()] if (P/'REPLACEMENTS.jsonl').exists() else []
h['CLI completed']+=sum(r.get('previous_status',{}).get('status')=='completed' for r in replaced)
h['terminal_failure_count']=sum(r['status'] in ['failed','lost','timed_out'] for r in rows)+sum(r.get('previous_status',{}).get('status') in ['failed','lost','timed_out'] for r in replaced)
h['termination_failure_rate']=h['terminal_failure_count']/(h['terminal_failure_count']+h['CLI completed']) if h['terminal_failure_count']+h['CLI completed'] else None
h['result_status_counts']=dict(collections.Counter(r['status'] for r in results))
h['result_rows']=len(results)
h['result_unique_keys']=len(set(tuple(r['key']) for r in results if r.get('key')))
h['stalled']=[r['worker'] for r in rows if r['heartbeat_age_s'] is not None and r['heartbeat_age_s']>240 and r['identity_alive']]
(P/'RESULT_INDEX.json').write_text(json.dumps({'utc':utc,'rows':results},ensure_ascii=False,indent=2))
(P/'GROUP_HEARTBEAT.json').write_text(json.dumps(h,indent=2,ensure_ascii=False))
with (P/'PROCESS_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(h,ensure_ascii=False)+'\n')
if not (P/'GROUP_STATUS.json').exists() or now-(P/'GROUP_STATUS.json').stat().st_mtime>280:(P/'GROUP_STATUS.json').write_text(json.dumps(h,indent=2,ensure_ascii=False))
if not (P/'PUBLISH_QUEUE.json').exists():(P/'PUBLISH_QUEUE.json').write_text(json.dumps({'utc':utc,'entries':[],'accepted_added':0,'reason':'Awaiting original full-checker evidence and deployable registration package'},indent=2))
print(json.dumps({k:h[k] for k in ['utc','active','alive','CLI completed','fullchecker passed','pending registry','root accepted','rolling_60m_root_accepted_added','result_rows','result_unique_keys','stalled','group_scope_now_root_accepted']},ensure_ascii=False))
for r in rows:
 hb=r.get('heartbeat') or {}
 print(r['worker'],r['status'],str(round(r['heartbeat_age_s'])) if r['heartbeat_age_s'] is not None else None,hb.get('phase'),hb.get('current_key',hb.get('key')),hb.get('blocker'))
