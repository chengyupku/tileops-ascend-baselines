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

def snapshot():
 now=time.time();plan=read(D/'PLAN.json');jobs=read(D/'JOBS.json',{'jobs':[]})['jobs'];con=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);workers=[];allrows={};flags=[]
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
  if not isinstance(rr,list):rr=[]
  allowed={(r['baseline'],r['operator'],r['case_id']) for r in read(wd/'SCOPE.json',[])}
  for r in rr:
   if not isinstance(r,dict):continue
   k=(r.get('baseline'),r.get('operator'),r.get('case_id'))
   if k in allowed and j.get('role')!='read_only_acceptance_audit':allrows[k]=r
  repeated=[{'count':n,'command':c[:400]} for c,n in collections.Counter(x['command'] for x in commands).items() if n>1]
  alive=live(dat.get('child_pid'),dat.get('child_stamp'));count=next(w['count'] for w in plan['workers'] if w['name']==j['name']);cmdtext='\n'.join(x['command'] for x in commands)
  w={'name':j['name'],'role':j.get('role','integration'),'host':j['host'],'scope':count,'id':jobid,'home':j['native']['home']['name'],'thread_id':thread,'native_status':status,'process_alive':alive,'effort':dat.get('effort'),'age_seconds':round(now-dat['created_at'],1),'events_age_seconds':round(now-ev.stat().st_mtime,1) if ev.exists() else None,'heartbeat':hb,'heartbeat_age_seconds':round(now-(wd/'HEARTBEAT.json').stat().st_mtime,1) if (wd/'HEARTBEAT.json').exists() else None,'result_counts':dict(collections.Counter(r.get('status') for r in rr if isinstance(r,dict))),'completed_commands':len(commands),'nonzero_commands':sum(x['exit_code'] not in (0,None) for x in commands),'repeated_command_flags':repeated,'last_commands':commands[-3:],'last_started_command':current[-1] if current else None,'guard_commands_observed':cmdtext.count('npu-guard run'),'api_queue_seconds':None}
  if status in ('running','starting') and not alive:flags.append(j['name']+': native active but child missing (check startup/supervisor before declaring lost)')
  if w['events_age_seconds'] and w['events_age_seconds']>600 and status=='running':flags.append(j['name']+': no event update >10 min; inspect active tool/queue')
  workers.append(w)
 counts=collections.Counter(r.get('status') for r in allrows.values());active=[w for w in workers if w['native_status'] in ('running','starting')];alive=sum(w['process_alive'] for w in active);elapsed=now-min(j['submitted_at'] for j in jobs) if jobs else 0
 accepted=read(D/'ACCEPTED.json',{'rows':[]}).get('rows',[])
 snap={'at':datetime.datetime.fromtimestamp(now,UTC).isoformat(),'elapsed_seconds':round(elapsed,1),'scope':700,'active_workers':len(active),'live_active_workers':alive,'observed_active_survival':alive/len(active) if active else None,'active_slot_occupancy':len(active)/8,'worker_reported_counts':dict(counts),'parent_accepted_completed':len(accepted),'parent_accepted_success_per_hour':len(accepted)*3600/elapsed if elapsed else None,'worker_reported_completed_per_hour':counts['completed']*3600/elapsed if elapsed else None,'workers':workers,'flags':flags,'notes':['CLI completed is not successful integration.','Repeated command flags require review; reads/status polling are not automatically waste.','API scheduling queue unavailable; device guard queues require real receipts.']}
 write(D/'LIVE.json',snap)
 with (D/'SNAPSHOTS.jsonl').open('a') as f:f.write(json.dumps(snap,ensure_ascii=False)+'\n')
 return snap

def hourly(snap):
 state=read(D/'STATE.json',{})
 if not state.get('enabled',False):return {'queued':False,'reason':'disabled'}
 msg=('[scheduled baseline700 hourly inspection] User authorized actual integration of 700 pairs on 2026-10-26, target100 successful pairs/h, medium workers and hourly inspection. Read '+str(D/'STATE.json')+' and LIVE.json, PLAN.json, JOBS.json. Respect any later pause. Inspect native worker process identity, receipts/guard queue and service health on all3 hosts; review successful agents commands for repeated unchanged failures, successful-case reruns, overbroad history reads, excess setup or CPU work while holding NPU. Audit completed result.txt plus full checker/original-boundary/formal registry evidence; only accept verified registered exact keys into ACCEPTED.json. Distinguish worker claimed success from accepted throughput and runtime success from CLI status. Diagnose biggest common bottleneck first and send scoped corrections. Refill completed/lost slots only with genuinely remaining700 scope via codex-workers existing manual-quota wrapper, gpt-6-astra medium, max8; avoid duplicate active ownership. Current scope corrected GDN94 from dense tolinear: PLAN/SCOPE authoritative; JOBS launch counts historical. Continue integration and hourly monitoring until all completed or user changes task; do not resurrect old HOLD queues. Record dated inspection and concrete actions in this report directory; update owned task memory. Snapshot: '+json.dumps({k:snap[k] for k in ['at','active_workers','live_active_workers','worker_reported_counts','parent_accepted_completed','flags']},ensure_ascii=False))
 t=time.time()
 try:
  p=subprocess.run(['/root/dyq/agent-bin/codex-7','queue','--thread',state['parent'],'--message',msg],capture_output=True,text=True,timeout=30);rc,out,err=p.returncode,p.stdout,p.stderr
 except (OSError,subprocess.TimeoutExpired) as e:rc,out,err=1,'',str(e)
 receipt={'at':datetime.datetime.now(UTC).isoformat(),'returncode':rc,'stdout':out,'stderr':err,'snapshot_at':snap['at'],'duration_seconds':time.time()-t,'message':msg}
 write(D/('HOURLY_'+datetime.datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'.json'),receipt)
 if rc:raise SystemExit(rc)
 return receipt

if __name__=='__main__':
 with (D/'MONITOR.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  s=snapshot()
  print(json.dumps({k:s[k] for k in ['at','active_workers','live_active_workers','worker_reported_counts','parent_accepted_completed','flags']},ensure_ascii=False))
  if '--hourly' in sys.argv:print(json.dumps(hourly(s),ensure_ascii=False))
