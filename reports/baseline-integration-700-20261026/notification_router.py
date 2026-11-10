#!/usr/bin/env python3
"""Route completion and monitoring metadata to delegated roots, never the controller."""
import datetime,fcntl,json,os,pathlib,sqlite3,subprocess,time
D=pathlib.Path(__file__).resolve().parent
DB='/root/dyq/codex-workers/workers.sqlite'
ACTIVE={'starting','running','stopping'}
def read(p,default=None):
 try:return json.loads(pathlib.Path(p).read_text())
 except (OSError,ValueError):return default
def live(j):
 try:
  s=pathlib.Path('/proc/%d/stat'%j['child_pid']).read_text().rsplit(')',1)[1].split()
  return s[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+s[19]==j['child_stamp']
 except (OSError,KeyError,TypeError):return False
def endpoint(role):
 q=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 t=next((t for t in q['tasks'] if t['name']==role),None)
 if not t or not t.get('job_id'):return None
 with sqlite3.connect('file:'+DB+'?mode=ro',uri=True) as c:
  r=c.execute('select data from jobs where id=?',(t['job_id'],)).fetchone()
 if not r:return None
 j=json.loads(r[0])
 if j['status'] not in ACTIVE or not live(j):return None
 thread=j.get('thread_id')
 if not thread:
  try:
   with (pathlib.Path(j['directory'])/'events.jsonl').open() as f:
    for n,line in enumerate(f):
     x=json.loads(line)
     if x.get('type')=='thread.started':thread=x['thread_id'];break
     if n>32:break
  except (OSError,ValueError):return None
 if not thread:return None
 state=read(D/'STATE.json',{})
 if thread==state.get('parent'):raise RuntimeError('Controller notification destination prohibited')
 return {'thread':thread,'home':j['home'],'job_id':j['id'],'role':role}
def group_endpoint(name):
 group=read(D/name/'GROUP.json',{})
 own=read(pathlib.Path(group.get('supervisor_self_job',D/name/'SELF_JOB.json')),{})
 if not own.get('id'):return None
 with sqlite3.connect('file:'+DB+'?mode=ro',uri=True) as db:
  row=db.execute('select data from jobs where id=?',(own['id'],)).fetchone()
 if not row:return None
 j=json.loads(row[0])
 if j.get('status') not in ACTIVE or not live(j):return None
 try:
  with (pathlib.Path(j['directory'])/'events.jsonl').open() as f:
   first=json.loads(f.readline())
  thread=first.get('thread_id')
 except (OSError,ValueError):return None
 if not thread or thread==read(D/'STATE.json',{}).get('parent'):return None
 return {'thread':thread,'home':j['home'],'job_id':j['id'],'role':'group:'+name}

def current_group_jobs():
 out={}
 for name in read(D/'STATE.json',{}).get('coordinator_groups',[]):
  for entry in read(D/name/'CHILD_JOBS.json',{'jobs':[]}).get('jobs',[]):
   ident=entry.get('native',{}).get('id')
   if ident:out[ident]=name
 return out

def sync_group_routes():
 """Keep native parent/quota intact; only the notification destination changes."""
 groups=current_group_jobs();parents=set();jobs=[]
 with sqlite3.connect('file:'+DB+'?mode=ro',uri=True) as db:
  for ident,name in groups.items():
   row=db.execute('select data from jobs where id=?',(ident,)).fetchone()
   if row:
    job=json.loads(row[0]);parents.add(job['parent']);jobs.append((name,job))
 routes=pathlib.Path('/root/dyq/codex-workers/notification_routes.json')
 with routes.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);cfg=read(routes,{'parents':{}})
  for parent in parents:cfg.setdefault('parents',{})[parent]={'resolver':str(D/'notification_router.py'),'controller_fallback':False,'authority':'Existing user instruction: completions follow corresponding active delegated root; native ownership unchanged'}
  tmp=routes.with_name(routes.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(cfg,indent=2)+'\n');os.replace(tmp,routes)
 return jobs,parents

def replay_misrouted_group_completions(jobs):
 """Durable metadata inbox; no worker result content is read or accepted here."""
 saved=read(D/'GROUP_COMPLETION_REROUTES.json',{'delivered_job_ids':[]});seen=set(saved['delivered_job_ids']);sent=0
 for name,job in jobs:
  if job.get('status') in ACTIVE:continue
  boxfile=D/name/'COMPLETION_MAILBOX.json';box=read(boxfile,{'entries':[]})
  if not any(e['job_id']==job['id'] for e in box['entries']):
   box['entries'].append({'job_id':job['id'],'status':job['status'],'result_path':str(pathlib.Path(job['directory'])/'result.txt'),'original_parent':job['parent'],'notice_target':job.get('notification_target'),'at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   tmp=boxfile.with_name(boxfile.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(box,indent=2)+'\n');os.replace(tmp,boxfile)
  target=group_endpoint(name)
  if not target or job.get('notification')!='queued' or job.get('notification_target',job['parent'])==target['thread'] or job['id'] in seen:continue
  msg=('[baseline700 repaired completion routing] Job '+job['id']+' status='+job['status']+' was queued to retired coordinator '+str(job.get('notification_target',job['parent']))+'. You are current group supervisor. Durable completion metadata: '+str(boxfile)+'. Check existing handling records before review; do not re-run or re-review already handled work. Process only unhandled assigned completions, route genuine publication packages to acceptance_root, record handled_job_ids in COMPLETION_HANDLED.json. This is notification metadata, not accepted success; controller did not inspect results.')
  result=subprocess.run(['/root/dyq/agent-bin/'+target['home']['name'],'queue','--thread',target['thread'],'--message',msg],capture_output=True,text=True,timeout=20)
  if result.returncode:continue
  seen.add(job['id']);sent+=1;saved['delivered_job_ids']=sorted(seen);saved['at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  tmp=(D/'GROUP_COMPLETION_REROUTES.json').with_suffix('.tmp');tmp.write_text(json.dumps(saved,indent=2)+'\n');os.replace(tmp,D/'GROUP_COMPLETION_REROUTES.json')
 return sent

def resolve_completion(job):
 group=current_group_jobs().get(job['id'])
 if group:return group_endpoint(group)
 role='acceptance_root'
 entry=next((x for x in read(D/'JOBS.json',{'jobs':[]})['jobs'] if x['native']['id']==job['id']),{})
 if entry.get('role')=='coordinator' or entry.get('name','').startswith('hour_'):role='dispatch_root'
 if entry.get('name')=='dispatch_root':role='acceptance_root'
 return endpoint(role)
def queue_role(role,message):
 target=endpoint(role)
 if not target:
  with (D/'NOTIFICATION_PENDING.jsonl').open('a') as f:f.write(json.dumps({'at':time.time(),'role':role,'message':message})+'\n')
  return subprocess.CompletedProcess([],75,'','Delegated root unavailable; saved durably; controller fallback forbidden')
 return subprocess.run(['/root/dyq/agent-bin/'+target['home']['name'],'queue','--thread',target['thread'],'--message',message],capture_output=True,text=True,timeout=25)
def drain_pending():
 """One unavailable role must not block other roles; retain durable per-line ACKs."""
 pendingfile=D/'NOTIFICATION_PENDING.jsonl'
 if not pendingfile.exists():return {'delivered':0,'pending':0}
 cursorfile=D/'NOTIFICATION_PENDING_CURSOR.json'
 saved=read(cursorfile,{'line':0});offset=saved.get('line',0)
 lines=pendingfile.read_text().splitlines();acked=set(saved.get('delivered_lines',[]));sent=0;began=time.monotonic()
 for i in range(offset,len(lines)):
  if i<offset or i in acked:continue
  if time.monotonic()-began>45:break
  x=json.loads(lines[i]);target=endpoint(x['role'])
  if not target:continue
  r=subprocess.run(['/root/dyq/agent-bin/'+target['home']['name'],'queue','--thread',target['thread'],'--message',x['message']],capture_output=True,text=True,timeout=25)
  if r.returncode:continue
  acked.add(i);sent+=1
  while offset in acked:acked.remove(offset);offset+=1
  temp=cursorfile.with_name(cursorfile.name+'.tmp.'+str(os.getpid()))
  temp.write_text(json.dumps({'line':offset,'delivered_lines':sorted(acked)})+'\n');os.replace(temp,cursorfile)
 return {'delivered':sent,'pending':len(lines)-offset-len(acked)}

def sweep():
 with (D/'NOTIFICATION_ROUTER.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  cfg=read(D/'NOTIFICATION_ROUTING.json',{});parent=read(D/'STATE.json',{})['parent'];count=0;pending=0
  groupjobs,parents=sync_group_routes();parents.add(parent)
  with sqlite3.connect('file:'+DB+'?mode=ro',uri=True) as c:rows=c.execute('select data from jobs where parent in ('+','.join('?' for _ in parents)+')',tuple(parents)).fetchall()
  group_replays=replay_misrouted_group_completions(groupjobs)
  for row in rows:
   j=json.loads(row[0])
   if j['status'] in ACTIVE or j.get('notification')=='queued':continue
   if not j.get('notification_route_pending') and j.get('created_at',0)<cfg.get('cutover_epoch',float('inf')):continue
   if resolve_completion(j) is None:pending+=1;continue
   r=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','notify',j['id']],capture_output=True,text=True,timeout=30)
   if r.returncode==0:
    delivered=json.loads(r.stdout);count+=delivered.get('notification')=='queued'
  durable=drain_pending()
  report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'delivered':count,'waiting_for_delegated_root':pending,'durable_pending':durable['pending'],'durable_delivered':durable['delivered'],'controller_notifications':0,'group_completion_replays':group_replays,'targets':{r:endpoint(r) for r in ['acceptance_root','dispatch_root']}}
  (D/'NOTIFICATION_ROUTER_LIVE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':sweep()
