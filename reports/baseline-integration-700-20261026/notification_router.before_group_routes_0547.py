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
def resolve_completion(job):
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
  with sqlite3.connect('file:'+DB+'?mode=ro',uri=True) as c:rows=c.execute('select data from jobs where parent=?',(parent,)).fetchall()
  for row in rows:
   j=json.loads(row[0])
   if j['status'] in ACTIVE or j.get('notification')=='queued':continue
   if not j.get('notification_route_pending') and j.get('created_at',0)<cfg.get('cutover_epoch',float('inf')):continue
   if resolve_completion(j) is None:pending+=1;continue
   r=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','notify',j['id']],capture_output=True,text=True,timeout=30)
   if r.returncode==0:
    delivered=json.loads(r.stdout);count+=delivered.get('notification')=='queued'
  durable=drain_pending()
  report={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'delivered':count,'waiting_for_delegated_root':pending,'durable_pending':durable['pending'],'durable_delivered':durable['delivered'],'controller_notifications':0,'targets':{r:endpoint(r) for r in ['acceptance_root','dispatch_root']}}
  (D/'NOTIFICATION_ROUTER_LIVE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':sweep()
