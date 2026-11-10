from notification_router import queue_role
#!/usr/bin/env python3
"""Read-only snapshots plus user-authorized hourly parent queue reminder."""
import collections,datetime,fcntl,json,os,pathlib,sqlite3,subprocess,sys,time
D=pathlib.Path(__file__).resolve().parent
UTC=datetime.timezone.utc

def read(p,default=None):
 try:return json.loads(pathlib.Path(p).read_text())
 except (OSError,ValueError):return default

def write(p,v):
 p=pathlib.Path(p);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));tmp.replace(p)

def live(pid,stamp):
 try:
  s=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();actual=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]
  return s[0]!='Z' and actual==stamp
 except (OSError,IndexError):return False

def exact_key(row):
 if not isinstance(row,dict):return None
 key=row.get('key',row)
 if isinstance(key,dict):key=[key.get(f) for f in ('baseline','operator','case_id')]
 if isinstance(key,(list,tuple)) and len(key)==3 and all(isinstance(v,str) and v for v in key):return tuple(key)
 return None

def snapshot():
 now=time.time();plan=read(D/'PLAN.json');jobs=read(D/'JOBS.json',{'jobs':[]})['jobs'];con=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);workers=[];allrows={};flags=[]
 root_jobs=list(jobs)
 for parent in root_jobs:
  if parent.get('role')=='coordinator':
   for child in read(pathlib.Path(parent['directory'])/'CHILD_JOBS.json',{'jobs':[]}).get('jobs',[]):
    if not any(x['native']['id']==child['native']['id'] for x in jobs):jobs.append({'host':'unknown','role':'delegated_integration',**child})
 for j in jobs:
  jobid=j['native']['id'];row=con.execute('select status,data from jobs where id=?',(jobid,)).fetchone();status,raw=row;dat=json.loads(raw);wd=pathlib.Path(j['directory']);ev=pathlib.Path(j['native']['directory'])/'events.jsonl';commands=[];current=[];thread=None
  if ev.exists():
   with ev.open() as f:
    for line in f:
     try:e=json.loads(line)
     except ValueError:continue
     if e.get('type')=='thread.started':thread=e.get('thread_id')
     item=e.get('item',{});cmd=item.get('command')
     if item.get('type')=='command_execution' and cmd:
      if e.get('type')=='item.completed':commands.append({'command':cmd,'exit_code':item.get('exit_code')})
      elif e.get('type')=='item.started':current.append(cmd)
  hb=read(wd/'HEARTBEAT.json',{});result=read(wd/'RESULT.json',{});rr=result if isinstance(result,list) else result.get('rows',result.get('results',result.get('cases',[])))
  if isinstance(result,dict) and not rr and result.get('status') and (result.get('key') or all(result.get(f) for f in ('baseline','operator','case_id'))):rr=[result]
  if not isinstance(rr,list):rr=[]
  scope_rows=read(wd/'SCOPE.json',[])
  if not isinstance(scope_rows,list):
   flags.append(j['name']+': malformed SCOPE; excluded from result aggregation');scope_rows=[]
  allowed={k for r in scope_rows if (k:=exact_key(r)) is not None}
  if len(allowed)!=len(scope_rows):flags.append(j['name']+': invalid or duplicate SCOPE keys; inspect exact ownership')
  for r in rr:
   if not isinstance(r,dict):continue
   rk=r.get('key')
   if isinstance(rk,(list,tuple)) and len(rk)==3:k=tuple(rk)
   elif isinstance(rk,dict):k=tuple(rk.get(f) for f in ('baseline','operator','case_id'))
   else:k=(r.get('baseline'),r.get('operator'),r.get('case_id'))
   if k in allowed and j.get('role','integration') in ('integration','scoped_integration_recovery','integration_diagnosis','cpu_registration','registration_audit','integration_threehost_failover','delegated_integration'):allrows[k]=r
  repeated=[{'count':n,'command':c[:400]} for c,n in collections.Counter(x['command'] for x in commands).items() if n>1]
  alive=live(dat.get('child_pid'),dat.get('child_stamp'));planned=next((w for w in plan['workers'] if w['name']==j['name']),None)
  count=planned['count'] if planned is not None else j.get('count')
  if planned is None and j.get('role')!='delegated_integration':flags.append(j['name']+': missing PLAN worker metadata; scope count from JOBS, reconcile dispatch records')
  cmdtext='\n'.join(x['command'] for x in commands)
  w={'name':j['name'],'role':j.get('role','integration'),'host':j.get('host','unknown'),'scope':count,'id':jobid,'home':j['native']['home']['name'],'thread_id':thread,'native_status':status,'process_alive':alive,'effort':dat.get('effort'),'age_seconds':round(now-dat['created_at'],1),'events_age_seconds':round(now-ev.stat().st_mtime,1) if ev.exists() else None,'heartbeat':hb,'heartbeat_age_seconds':round(now-(wd/'HEARTBEAT.json').stat().st_mtime,1) if (wd/'HEARTBEAT.json').exists() else None,'result_counts':dict(collections.Counter(r.get('status') for r in rr if isinstance(r,dict))),'completed_commands':len(commands),'nonzero_commands':sum(x['exit_code'] not in (0,None) for x in commands),'repeated_command_flags':repeated,'last_commands':commands[-3:],'last_started_command':current[-1] if current else None,'guard_commands_observed':cmdtext.count('npu-guard run'),'api_queue_seconds':None}
  if status in ('running','starting') and not alive:flags.append(j['name']+': native active but child missing (check startup/supervisor before declaring lost)')
  if w['events_age_seconds'] and w['events_age_seconds']>600 and status=='running':flags.append(j['name']+': no event update >10 min; inspect active tool/queue')
  workers.append(w)
 counts=collections.Counter(r.get('status') for r in allrows.values());active=[w for w in workers if w['native_status'] in ('running','starting')];alive=sum(w['process_alive'] for w in active);elapsed=now-min(j['submitted_at'] for j in jobs) if jobs else 0
 accepted=read(D/'ACCEPTED.json',{'rows':[]}).get('rows',[])
 accepted_last_hour=sum(now-3600 <= datetime.datetime.fromisoformat(r['at']).timestamp() <= now for r in accepted if r.get('at'))
 execution=[w for w in active if w['role']!='coordinator'];coordinators=[w for w in active if w['role']=='coordinator']
 target=read(D/'STATE.json',{}).get('target_execution_workers',8)
 if len(accepted)<700 and len(execution)<target:flags.append(f'execution worker slots underfilled: {len(execution)}/{target}; inspect genuinely actionable scope and scripted refill; not evidence of worker death')
 state=read(D/'STATE.json',{});required_rate=state.get('target_success_per_hour',100)
 if state.get('deadline_utc'):
  seconds_left=datetime.datetime.fromisoformat(state['deadline_utc']).timestamp()-now
  required_rate=(700-len(accepted))*3600/seconds_left if seconds_left>0 else None
  if seconds_left<=0 and len(accepted)<700:flags.append('User deadline reached with unaccepted scope; report exact remaining and blockers, do not reset clock')
 if elapsed>=3600 and required_rate is not None and accepted_last_hour<required_rate and len(accepted)<700:flags.append(f'accepted throughput below deadline requirement: {accepted_last_hour}/h versus{required_rate:.1f}/h')
 snap={'required_accepted_per_hour':required_rate,'deadline_utc':state.get('deadline_utc'),'accepted_last_hour':accepted_last_hour,'at':datetime.datetime.fromtimestamp(now,UTC).isoformat(),'elapsed_seconds':round(elapsed,1),'scope':700,'active_workers':len(active),'live_active_workers':alive,'observed_active_survival':alive/len(active) if active else None,'active_slot_occupancy':len(execution)/target,'execution_workers':len(execution),'execution_workers_alive':sum(w['process_alive'] for w in execution),'coordinator_workers':len(coordinators),'target_execution_workers':target,'worker_reported_counts':dict(counts),'parent_accepted_completed':len(accepted),'parent_accepted_success_per_hour':len(accepted)*3600/elapsed if elapsed else None,'worker_reported_completed_per_hour':counts['completed']*3600/elapsed if elapsed else None,'workers':workers,'flags':flags,'notes':['CLI completed is not successful integration.','Repeated command flags require review; reads/status polling are not automatically waste.','API scheduling queue unavailable; device guard queues require real receipts.']}
 write(D/'LIVE.json',snap)
 with (D/'SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(snap,ensure_ascii=False)+'\n')
 return snap

def periodic_delivery_blocked(state):
 # No atomic idle-only enqueue is available. Keep observations on disk instead.
 return bool(state.get('controller_interrupt_inspection_forbidden'))

def hourly(snap):
 state=read(D/'STATE.json',{})
 if not state.get('enabled',False):return {'queued':False,'reason':'disabled'}
 if periodic_delivery_blocked(state):return {'queued':False,'reason':'idle-only policy; no reliable idle enqueue available'}
 msg=('[scheduled baseline700 hourly inspection] User authorized actual integration of 700 pairs on 2026-10-26, deadlineUTC2026-10-27T05:31:00 for ALL remaining integration; required throughput=remaining unaccepted/hours left, not fixed100/h. Medium workers and continuous inspection. Read '+str(D/'STATE.json')+' and LIVE.json, PLAN.json, JOBS.json. Respect any later pause. Mandatory user correction20261027 applies to ALL tasks: if one host has external NPU occupation, inspect other two hosts and migrate compatible remaining work; do not unchanged-retry on the occupied host then abandon. Use THREEHOST_FAILOVER_POLICY_20261027.json and current mutually exclusive PLAN ownership; preserve original implementation/checker and host guard. Inspect native worker process identity, receipts/guard queue and service health on all3 hosts; review successful agents commands for repeated unchanged failures, successful-case reruns, overbroad history reads, excess setup or CPU work while holding NPU. Per-worker evidence review and registration belong to acceptance_root; notify that root to drain its queues. Do not write ACCEPTED.json as dispatch_root. Distinguish worker claimed success from accepted throughput and runtime success from CLI status. Diagnose biggest common bottleneck first and send scoped corrections. Refill completed/lost slots only with genuinely remaining700 scope via codex-workers existing manual-quota wrapper, gpt-6-astra medium; native max8 per real parent, current fleet target25 execution workers under3 coordinators, user requires >=24 and up to48 when needed. Use scripted refill from validated exclusive ready queue; root does not routinely manually spawn children. Monitor all descendants; avoid duplicate ownership. DeadlineUTC2026-10-27T05:31:00 is not reset by restarts. Current scope corrected GDN94 from dense tolinear: PLAN/SCOPE authoritative; JOBS launch counts historical. Continue integration and hourly monitoring until all completed or user changes task; do not resurrect old HOLD queues. Record dated inspection and concrete actions in this report directory; update owned task memory. Snapshot: '+json.dumps({k:snap[k] for k in ['at','active_workers','live_active_workers','worker_reported_counts','parent_accepted_completed','required_accepted_per_hour','flags']},ensure_ascii=False))
 t=time.time()
 try:
  p=queue_role('dispatch_root',msg);rc,out,err=p.returncode,p.stdout,p.stderr
 except (OSError,subprocess.TimeoutExpired) as e:rc,out,err=1,'',str(e)
 receipt={'at':datetime.datetime.now(UTC).isoformat(),'returncode':rc,'stdout':out,'stderr':err,'snapshot_at':snap['at'],'duration_seconds':time.time()-t,'message':msg}
 write(D/('HOURLY_'+datetime.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),receipt)
 if rc:raise SystemExit(rc)
 return receipt

def fleet_intervention(snap):
 state=read(D/'STATE.json',{})
 if not state.get('enabled') or not state.get('coordinator_groups'):return
 if periodic_delivery_blocked(state):return
 now=time.time();last=read(D/'FLEET_INTERVENTION.json',{})
 urgent=snap['live_active_workers']<snap['active_workers']
 if now-last.get('notified_at_epoch',0)<(120 if urgent else 300):return
 queues={name:bool(read(D/name/'PUBLISH_QUEUE.json',{}).get('entries',[])) for name in state['coordinator_groups']}
 summary={k:snap.get(k) for k in ['at','execution_workers','execution_workers_alive','coordinator_workers','accepted_last_hour','parent_accepted_completed','required_accepted_per_hour','flags']}
 msg='[baseline700 active fleet inspection] User requires >=24 execution agents (coordinators excluded), up to48 if useful, balanced three-host work, scripted refill, no wasted retries. Deadline '+state.get('deadline_utc','unknown')+'. Respect later user pause. Inspect refill_fleet service, child READY_QUEUE and native identities; have acceptance_root process concrete PUBLISH_QUEUE and correct the biggest dispatch waste/queue bottleneck. Do not merely repeat a status message. Snapshot '+json.dumps(summary,ensure_ascii=False)+'; group publish queues '+json.dumps(queues)
 r=queue_role('dispatch_root',msg)
 record={'at':datetime.datetime.now(UTC).isoformat(),'returncode':r.returncode,'summary':summary,'publish_queues':queues,'notified_at_epoch':now if r.returncode==0 else last.get('notified_at_epoch',0)}
 write(D/'FLEET_INTERVENTION.json',record)

def monitor_fault(error):
 # A timer remaining active must never conceal failed monitoring.
 now=time.time();last=read(D/'MONITOR_FAULT.json',{});state=read(D/'STATE.json',{})
 fault={'at':datetime.datetime.now(UTC).isoformat(),'error':type(error).__name__+': '+str(error),'last_snapshot_at':read(D/'LIVE.json',{}).get('at'),'last_notified_at':last.get('last_notified_at',0)}
 write(D/'MONITOR_FAULT.json',fault)
 if state.get('enabled') and not periodic_delivery_blocked(state) and now-fault['last_notified_at']>=600:
  msg='[baseline700 monitor failure] Sampling or hourly inspection failed: '+fault['error']+'. Latest snapshot '+str(fault['last_snapshot_at'])+'. Inspect service and repair before trusting liveness/queue/throughput. Respect latest user pause; this is not authorization to launch work.'
  try:
   r=queue_role('dispatch_root',msg)
   fault['notification_returncode']=r.returncode
   if r.returncode==0:fault['last_notified_at']=now
  except (OSError,subprocess.TimeoutExpired) as e:fault['notification_error']=str(e)
  write(D/'MONITOR_FAULT.json',fault)

if __name__=='__main__':
 with (D/'MONITOR.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  try:
   s=snapshot()
   print(json.dumps({k:s[k] for k in ['at','active_workers','live_active_workers','worker_reported_counts','parent_accepted_completed','required_accepted_per_hour','flags']},ensure_ascii=False))
   fleet_intervention(s)
   if '--hourly' in sys.argv:print(json.dumps(hourly(s),ensure_ascii=False))
  except Exception as error:
   monitor_fault(error)
   raise
