#!/usr/bin/env python3
"""Native supervised refill from coordinator-approved exclusive ready queues."""
import datetime,fcntl,json,os,pathlib,sqlite3,subprocess,time
D=pathlib.Path(__file__).resolve().parent;W=D.parent.parent
ACTIVE={'starting','running','stopping'}
def read(p,default=None):
 try:return json.loads(pathlib.Path(p).read_text())
 except (OSError,ValueError):return default

def put(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def key(r):
 if not isinstance(r,dict):return None
 x=r.get('key',r.get('exactkey',r))
 if isinstance(x,dict):x=[x.get(k) for k in ('baseline','operator','case_id')]
 return tuple(x) if isinstance(x,(list,tuple)) and len(x)==3 and all(isinstance(v,str) and v for v in x) else None

def live(pid,stamp):
 try:
  v=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return v[0]!='Z' and pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()+':'+v[19]==stamp
 except (OSError,IndexError):return False

def validated_keys(obj):
 out=set()
 if isinstance(obj,dict):
  k=key(obj);status=obj.get('status',obj.get('checker_status'))
  if k and status in {'verified','full_checker_native_verified','completed_fullchecker_verified','verified_pending_parent_review','verified_pending_parent_acceptance','verified_registered','numerically_verified_pending_parent_review'}:out.add(k)
  for v in obj.values():out|=validated_keys(v)
 elif isinstance(obj,list):
  for v in obj:out|=validated_keys(v)
 return out

def eligibility(entry,p,group_keys,owned,passed,db):
 if not entry.get('ready') or not entry.get('approved_by_coordinator'):return 'not coordinator approved'
 q=pathlib.Path(entry['directory']).resolve()
 if not q.is_relative_to(p) or q==p:return 'outside group write scope'
 rows=read(q/'SCOPE.json',[]);keys={key(r) for r in rows}
 if not keys or None in keys or len(keys)!=len(rows) or not keys<=group_keys:return 'invalid scope'
 if keys & owned:return 'duplicate active owner'
 if keys & passed:return 'already passed or accepted: remove from execution queue'
 if not (q/'TASK.txt').is_file():return 'missing concrete task'
 for jid in entry.get('predecessor_jobs',[]):
  row=db.execute('select status,data from jobs where id=?',(jid,)).fetchone()
  if row is None:return 'unknown predecessor'
  dat=json.loads(row[1])
  if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):return 'predecessor still active'
 if entry.get('predecessor_jobs') and not entry.get('previous_devices_quiescent'):return 'predecessor device quiescence unconfirmed'
 if entry.get('retry') and not entry.get('changed_condition'):return 'retry lacks changed condition'
 return None

def dispatch_controls(state,db,deadline,record):
 """Script-dispatch explicitly approved root coordination roles, never pair owners."""
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 for entry in queue['tasks']:
  if not entry.get('ready') or entry.get('job_id'):continue
  q=pathlib.Path(entry['directory']).resolve()
  if not q.is_relative_to(D) or not (q/'TASK.txt').is_file() or entry.get('role')!='coordinator':
   record['rejected'].append({'control':entry.get('name'),'reason':'invalid approved coordination task'});continue
  active=0
  for status,data in db.execute('select status,data from jobs where parent=?',(state['parent'],)):
   j=json.loads(data);active+=status in ACTIVE or live(j.get('child_pid'),j.get('child_stamp'))
  if active>=8:break
  cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--model','gpt-6-astra','--effort','medium','--timeout',str(max(60,min(5400,int(deadline-time.time())))),'--notify','queue','--sandbox','danger-full-access','--parent',state['parent'],'--parent-home','codex-7']
  for path in entry['write']:cmd+=['--write',path]
  began=time.time();r=subprocess.run(cmd,capture_output=True,text=True);put(q/'SPAWN.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if r.returncode:record['rejected'].append({'control':entry['name'],'reason':'native spawn failed','stderr':r.stderr[-1000:]});continue
  native=json.loads(r.stdout);entry['job_id']=native['id'];entry['ready']=False;put(q/'SELF_JOB.json',native);put(D/'ROOT_CONTROL_QUEUE.json',queue)
  with (D/'DISPATCH.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);jobs=read(D/'JOBS.json',{'jobs':[]});jobs['jobs'].append({'name':entry['name'],'role':'coordinator','host':'CPU','count':0,'directory':str(q),'task_file':str(q/'TASK.txt'),'native':native,'submitted_at':began,'returned_at':time.time(),'active_assignment':True});put(D/'JOBS.json',jobs)
  record['launched'].append({'name':entry['name'],'id':native['id'],'role':'coordinator','pair_ownership':0})

def main():
 with (D/'FLEET_REFILL.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state=read(D/'STATE.json',{});now=time.time();record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launched':[],'rejected':[],'groups':[]}
  if not state.get('enabled') or state.get('phase') in ('paused','stopped','complete'):return
  deadline=datetime.datetime.fromisoformat(state['deadline_utc']).timestamp()
  if now>=deadline:record['stopped']='deadline';put(D/'FLEET_REFILL_LIVE.json',record);return
  db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
  dispatch_controls(state,db,deadline,record)
  owned=set();passed={tuple(r['key']) for r in read(D/'ACCEPTED.json',{'rows':[]})['rows']}
  groups=[D/n for n in state.get('coordinator_groups',[])]
  alljobs=list(read(D/'JOBS.json',{'jobs':[]})['jobs'])
  for p in groups:alljobs+=read(p/'CHILD_JOBS.json',{'jobs':[]})['jobs']
  for j in alljobs:
   if j.get('role')=='coordinator':continue
   row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone()
   if not row:continue
   dat=json.loads(row[1]);q=pathlib.Path(j['directory'])
   if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):owned|={key(r) for r in read(q/'SCOPE.json',[])}
   else:passed|=validated_keys(read(q/'RESULT.json',{}))
  for p in groups:
   with (p/'CHILD_DISPATCH.lock').open('a') as childlock:
    fcntl.flock(childlock,fcntl.LOCK_EX);jobs=read(p/'CHILD_JOBS.json',{'jobs':[]})['jobs'];selfjob=read(p/'SELF_JOB.json');parentrow=db.execute('select status,data from jobs where id=?',(selfjob['id'],)).fetchone()
    if not parentrow or parentrow[0] not in ACTIVE:record['rejected'].append({'group':p.name,'reason':'coordinator inactive; root must restore supervision'});continue
    active=0
    for j in jobs:
     row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();d=json.loads(row[1])
     active+=row[0] in ACTIVE or live(d.get('child_pid'),d.get('child_stamp'))
    group=read(p/'GROUP.json');limit=group.get('children',7);groupkeys={key(r) for r in read(p/'SCOPE.json',[])};queue=read(p/'READY_QUEUE.json',{'tasks':[]})['tasks'];record['groups'].append({'name':p.name,'active':active,'limit':limit,'ready_entries':sum(bool(x.get('ready')) for x in queue)})
    for entry in queue:
     if active>=limit:break
     q=pathlib.Path(entry['directory']).resolve()
     if any(j['directory']==str(q) for j in jobs):continue
     reason=eligibility(entry,p,groupkeys,owned,passed,db)
     if reason:record['rejected'].append({'group':p.name,'directory':str(q),'reason':reason});continue
     tid=None
     for line in (pathlib.Path(selfjob['directory'])/'events.jsonl').open():
      e=json.loads(line)
      if e.get('type')=='thread.started':tid=e['thread_id'];break
     if not tid:record['rejected'].append({'group':p.name,'reason':'no native coordinator thread'});break
     owner='baseline700-20261026/'+p.name+'/'+q.name
     cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--write',str(q),'--model','gpt-6-astra','--effort','medium','--timeout',str(max(60,min(3600,int(deadline-time.time())))),'--notify','queue','--sandbox','danger-full-access','--parent',tid,'--parent-home',selfjob['home']['name']]
     if entry.get('allow_npu',True):cmd+=['--npu-owner',owner,'--npu-timeout',str(min(600,entry.get('npu_timeout',600))),'--npu-wait','120']
     began=time.time();r=subprocess.run(cmd,capture_output=True,text=True);put(q/'SPAWN.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
     if r.returncode:record['rejected'].append({'group':p.name,'directory':str(q),'reason':'native spawn failed','stderr':r.stderr[-1000:]});break
     native=json.loads(r.stdout);keys={key(r) for r in read(q/'SCOPE.json')};row={'name':p.name+'/'+q.name,'host':'compatible_threehost','count':len(keys),'directory':str(q),'task_file':str(q/'TASK.txt'),'native':native,'submitted_at':began,'returned_at':time.time(),'role':'delegated_integration','coordinator':p.name,'parent_thread':tid,'active_assignment':True};jobs.append(row);put(p/'CHILD_JOBS.json',{'jobs':jobs});owned|=keys;active+=1;record['launched'].append({'name':row['name'],'id':native['id'],'count':len(keys)})
  put(D/'FLEET_REFILL_LIVE.json',record)
  with (D/'FLEET_REFILL_HISTORY.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
