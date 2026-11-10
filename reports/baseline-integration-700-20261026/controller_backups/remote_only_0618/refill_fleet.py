#!/usr/bin/env python3
"""Native supervised refill from coordinator-approved exclusive ready queues."""
import datetime,fcntl,json,os,pathlib,sqlite3,subprocess,time,hashlib
D=pathlib.Path(__file__).resolve().parent;W=D.parent.parent
ACTIVE={'starting','running','stopping'}
def read(p,default=None):
 try:return json.loads(pathlib.Path(p).read_text())
 except (OSError,ValueError):return default

def put(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
def key(r):
 if not isinstance(r,dict):return None
 x=r.get('key',r.get('exactkey',r.get('exact_key',r)))
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
  k=key(obj);status=obj.get('status') or obj.get('checker_status')
  if k and status in {'verified','full_checker_native_verified','completed_fullchecker_verified','verified_pending_parent_review','verified_pending_parent_acceptance','verified_registered','numerically_verified_pending_parent_review','verified_pending_parent_registration','verified_ready_for_parent_serial_publication','verified_pending_publication','numerically_verified_pending_registration'}:out.add(k)
  checker=obj.get('checker')
  if k and (obj.get('checker_status')=='verified' or isinstance(checker,dict) and checker.get('status')=='verified'):out.add(k)
  for v in obj.values():out|=validated_keys(v)
 elif isinstance(obj,list):
  for v in obj:out|=validated_keys(v)
 return out

def eligibility(entry,p,group_keys,owned,passed,db):
 if entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete'):return 'entry disabled'
 if not entry.get('ready') or not entry.get('approved_by_coordinator'):return 'not coordinator approved'
 q=pathlib.Path(entry['directory']).resolve()
 if not q.is_relative_to(p) or q==p:return 'outside group write scope'
 rows=read(q/'SCOPE.json',[]);keys={key(r) for r in rows}
 if not keys or None in keys or len(keys)!=len(rows) or not keys<=group_keys:return 'invalid scope'
 if keys & owned:return 'duplicate active owner'
 if keys & passed:return 'already passed or accepted: remove from execution queue'
 if not (q/'TASK.txt').is_file():return 'missing concrete task'
 if read(D/'STATE.json',{}).get('require_explicit_host_assignment') and entry.get('allow_npu',True):
  assignment=read(q/'HOST_ASSIGNMENT.json',{})
  if assignment.get('host') not in {'local','openlibing-DevEnv-055409','openlibing-DevEnv-661495'} or not assignment.get('approved_by_coordinator') or assignment.get('scope_keys')!=sorted([list(k) for k in keys]):return 'missing approved exact-scope host assignment'
 for jid in entry.get('predecessor_jobs',[]):
  row=db.execute('select status,data from jobs where id=?',(jid,)).fetchone()
  if row is None:return 'unknown predecessor'
  dat=json.loads(row[1])
  if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):return 'predecessor still active'
 if entry.get('predecessor_jobs') and not entry.get('previous_devices_quiescent'):return 'predecessor device quiescence unconfirmed'
 if entry.get('retry') and not entry.get('changed_condition'):return 'retry lacks changed condition'
 return None

def dispatch_open(deadline=None):
 """Delivery SLA is never reset and is not an implicit stop authorization."""
 s=read(D/'STATE.json',{})
 if not s.get('enabled') or s.get('phase') in ('paused','stopped','complete'):return False
 stop=s.get('dispatch_stop_at')
 return not stop or time.time()<datetime.datetime.fromisoformat(stop).timestamp()-60

def native_budget_seconds(maximum, allow_unbounded=False):
 s=read(D/'STATE.json',{});stop=s.get('dispatch_stop_at')
 if allow_unbounded and s.get('execution_wall_timeout_disabled') and not stop:return 0
 return max(60,min(maximum,int(datetime.datetime.fromisoformat(stop).timestamp()-time.time()))) if stop else maximum

def dispatch_controls(state,db,deadline,record):
 """Script-dispatch explicitly approved root coordination roles, never pair owners."""
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 for entry in queue['tasks']:
  if not dispatch_open(deadline):record['stopped']='disabled or explicit hard-stop margin';break
  if entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete'):continue
  if not entry.get('ready') or entry.get('job_id'):continue
  q=pathlib.Path(entry['directory']).resolve()
  if not q.is_relative_to(D) or not (q/'TASK.txt').is_file() or entry.get('role')!='coordinator':
   record['rejected'].append({'control':entry.get('name'),'reason':'invalid approved coordination task'});continue
  # Apply the current compact coordination contract before native launch.
  contract_path=D/'COORDINATOR_CURRENT_CONTRACT.txt'
  if entry.get('restores_group') and contract_path.exists():
   group_path=pathlib.Path(entry['restores_group'])
   compact=contract_path.read_text()+'\n你的角色是组协调者，接管目录'+str(group_path)+'；当前角色目录'+str(q)+'。保持现有children真实父级与所有权；GROUP.supervisor_self_job必须指向本角色目录的SELF_JOB.json，不引用前任路径。\n'
   put(q/'COMPACT_CONTRACT_APPLIED.json',{'version':'20261027T0558','at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
   (q/'TASK.txt').write_text(compact)
  dispatch_parent=state['parent'];dispatch_home='codex-7'
  if entry.get('parent_self_job'):
   ps=read(entry['parent_self_job']);pr=db.execute('select status,data from jobs where id=?',(ps['id'],)).fetchone()
   if not pr or pr[0] not in {'starting','running'}:record['rejected'].append({'control':entry['name'],'reason':'requested real parent inactive'});continue
   dispatch_home=ps['home']['name'];dispatch_parent=None
   for line in (pathlib.Path(ps['directory'])/'events.jsonl').open():
    e=json.loads(line)
    if e.get('type')=='thread.started':dispatch_parent=e['thread_id'];break
   if not dispatch_parent:record['rejected'].append({'control':entry['name'],'reason':'no real parent thread'});continue
  active=0
  for status,data in db.execute('select status,data from jobs where parent=?',(dispatch_parent,)):
   j=json.loads(data);active+=status in ACTIVE or live(j.get('child_pid'),j.get('child_stamp'))
  if active>=8:break
  cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--model','gpt-6-astra','--effort','medium','--timeout',str(native_budget_seconds(5400)),'--notify','queue','--sandbox','danger-full-access','--parent',dispatch_parent,'--parent-home',dispatch_home]
  for path in entry['write']:cmd+=['--write',path]
  began=time.time();r=subprocess.run(cmd,capture_output=True,text=True);put(q/'SPAWN.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
  if r.returncode:record['rejected'].append({'control':entry['name'],'reason':'native spawn failed','stderr':r.stderr[-1000:]});continue
  native=json.loads(r.stdout);entry['job_id']=native['id'];entry['ready']=False;put(q/'SELF_JOB.json',native);put(D/'ROOT_CONTROL_QUEUE.json',queue)
  with (D/'DISPATCH.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);jobs=read(D/'JOBS.json',{'jobs':[]});jobs['jobs'].append({'name':entry['name'],'role':'coordinator','host':'CPU','count':0,'directory':str(q),'task_file':str(q/'TASK.txt'),'native':native,'submitted_at':began,'returned_at':time.time(),'active_assignment':True});put(D/'JOBS.json',jobs)
   plan=read(D/'PLAN.json',{'workers':[]})
   if not any(w.get('name')==entry['name'] for w in plan['workers']):
    plan['workers'].append({'name':entry['name'],'role':'coordinator','host':'CPU','count':0,'directory':str(q),'task_file':str(q/'TASK.txt'),'active_assignment':True});put(D/'PLAN.json',plan)
  record['launched'].append({'name':entry['name'],'id':native['id'],'role':'coordinator','pair_ownership':0})

def restore_supervision(p,old,group,state,db,deadline,record):
 """Queue a real coordinator only for surviving children or eligible ready work."""
 name='supervise_'+p.name+'_'+old['id'][:8]
 q=D/'coordinator_replacements'/name;q.mkdir(parents=True,exist_ok=True)
 task=q/'TASK.txt'
 if not task.exists():
  task.write_text(f"""你是恢复监督的真实coordinator，gpt-6-astra medium。首先读AGENTS.md、MEMORY索引和codex-workers SKILL，只读必要当前记忆。交付目标{state['deadline_utc']}不重置；逾期如实记账但不自动停工，只有明确暂停/硬停/无合法工作才停派。你自己禁止NPU；允许原生manual-quota wrapper + medium + --notify queue派实际执行子worker，每父最多8，所有设备动作前台host npu-guard。
接管监督目录{p}，旧coordinator已终态，既有CHILD_JOBS中的实际执行子worker所有权保持原样，不能取消或重复派遣，不能伪造parent。refill将GROUP.supervisor_self_job切到你的真实SELF_JOB，所有后续新子任务走READY_QUEUE/refill_fleet，现有活跃children计入组上限。
只做必要任务分工、CPU准备及补位，禁止写ACCEPTED/providers/memory。用户当前仅控制层汇总，数值验收交acceptance_root真实线程。读BASELINE_WORKER_PROMPT_RULES_20261027.txt。按host兼容与预计分钟均衡，不以代表失败跳过未跑exactkey；无changed condition不重复失败；已完整成功不再跑。原kernel/compiler/pass/schedule/reference/seed/tolerance不改。
先核现有children PID+启动身份和READY_QUEUE；只在旧owner完成释放、进程设备任务停止后转移范围。CPU准备在guard外，外部占用核另外两host并迁移兼容任务，不杀外部进程或改锁。你获准写报告目录下本组分工/队列/monitor和私有任务；STATE/PLAN/JOBS持DISPATCH.lock，CHILD_JOBS持CHILD_DISPATCH.lock，FLEET变更持FLEET_REFILL.lock。每2分钟HEARTBEAT，每5分钟措施；实在无合法可执行工作则写DISPATCH_HANDOFF，别为凑24制造任务。持续至职责完成/用户暂停或明确硬停；交付目标到期不等于自动停工。完成通知由native supervisor处理，勿主动queue顶层主会话。
""",encoding='utf-8')
  put(q/'SCOPE.json',[])
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 if not any(e.get('name')==name for e in queue['tasks']):
  entry={'name':name,'role':'coordinator','directory':str(q),'ready':True,'write':[str(D)],'restores_group':str(p)}
  parent=read(D/'dispatch_root/SELF_JOB.json')
  if parent:
   row=db.execute('select status from jobs where id=?',(parent['id'],)).fetchone()
   if row and row[0] in {'starting','running'}:entry['parent_self_job']=str(D/'dispatch_root/SELF_JOB.json')
  queue['tasks'].append(entry);put(D/'ROOT_CONTROL_QUEUE.json',queue)
 group['previous_supervisor_self_job']=group.get('supervisor_self_job',str(p/'SELF_JOB.json'))
 group['supervisor_self_job']=str(q/'SELF_JOB.json');put(p/'GROUP.json',group)
 record['rejected'].append({'group':p.name,'reason':'real supervisor replacement queued; existing child ownership retained','replacement':name})

def wake_acceptance_for_new_publish(state,db,record):
 """Wake the authorized acceptance receiver only for new evidence.

 A queued launch is not consumption.  ``publish_wake_pending`` survives a
 receiver crash; a disposition/ACK must identify both the wake signature and
 the candidate digests it handled.  Rejected candidates therefore do not
 retrigger forever, while changed evidence receives a new digest.
 """
 # Re-check the live task boundary before any receiver mutation.
 live_state=read(D/'STATE.json',state)
 if not live_state.get('enabled') or live_state.get('phase') in ('paused','stopped','complete'):return
 deadline_value=live_state.get('deadline_utc')
 if not deadline_value or not dispatch_open(datetime.datetime.fromisoformat(deadline_value).timestamp()):return
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 entry=next((e for e in queue['tasks'] if e.get('name')=='acceptance_root'),None)
 if not entry or entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete'):return
 now=time.time()
 disposition=read(D/'acceptance_root'/'WAKE_DISPOSITION.json',{})
 ack_record=read(D/'acceptance_root'/'WAKE_ACK.json',{})
 acked_signatures=set(entry.get('publish_wake_delivered_signatures',[]))
 handled=set(entry.get('publish_wake_handled_digests',[]))
 for item in (disposition,ack_record):
  for k in ('processed_signatures','ack_signatures','handled_signatures'):
   vals=item.get(k,[]); acked_signatures.update(vals if isinstance(vals,list) else [vals])
  for k in ('processed_candidate_digests','candidate_digests','handled_candidate_digests'):
   vals=item.get(k,[]); handled.update(vals if isinstance(vals,list) else [vals])
  for k in ('processed_signature','ack_signature'):
   if item.get(k):acked_signatures.add(item[k])
  for k in ('processed_candidate_digest','candidate_digest'):
   if item.get(k):handled.add(item[k])
 pending=list(entry.get('publish_wake_pending',[]))
 remaining=[]
 for item in pending:
  digs=set(item.get('candidate_digests',[]))
  if item.get('signature') in acked_signatures or (digs and digs <= handled):
   acked_signatures.add(item.get('signature'))
   handled.update(digs)
   record.setdefault('wake_acks',[]).append({'signature':item.get('signature'),'candidate_digests':sorted(digs),'status':'processed_ack'})
  else: remaining.append(item)
 entry['publish_wake_pending']=remaining
 entry['publish_wake_delivered_signatures']=sorted(x for x in acked_signatures if x)
 entry['publish_wake_handled_digests']=sorted(x for x in handled if x)
 put(D/'ROOT_CONTROL_QUEUE.json',queue)
 # A dead receiver may be retried, but never more often than the backoff.
 if remaining:
  row=None
  if entry.get('job_id'): row=db.execute('select status,data from jobs where id=?',(entry['job_id'],)).fetchone()
  dead=not row
  if row:
   dat=json.loads(row[1]); dead=not (row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')))
  retry_at=float(entry.get('publish_wake_retry_at',0))
  if dead and now >= retry_at and entry.get('ready') is False:
   old=entry.pop('job_id',None)
   if old: entry.setdefault('prior_job_ids',[]).append(old)
   entry['ready']=True; entry['publish_wake_retry_at']=now+120
   entry['changed_condition']='Acceptance receiver died before disposition; replay durable pending wake without changing owner'
   put(D/'ROOT_CONTROL_QUEUE.json',queue)
   put(D/'acceptance_root'/'WAKE_REQUEST.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'signature':remaining[-1].get('signature'),'status':'pending_receiver_ack','retry':True,'ack_file':str(D/'acceptance_root'/'WAKE_DISPOSITION.json')})
   record.setdefault('wake_retries',[]).append({'pending':len(remaining),'backoff_seconds':120})
  else: record.setdefault('wake_pending',[]).extend(remaining)
  return
 if entry.get('ready') or not entry.get('job_id'):return
 row=db.execute('select status,data from jobs where id=?',(entry['job_id'],)).fetchone()
 if not row:return
 dat=json.loads(row[1])
 if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):return
 accepted={tuple(r['key']) for r in read(D/'ACCEPTED.json',{'rows':[]})['rows']}
 candidates=[]
 for name in state.get('coordinator_groups',[]):
  path=D/name/'PUBLISH_QUEUE.json'
  for item in read(path,{'entries':[]}).get('entries',[]):
   k=key(item)
   if not k or k in accepted or not (item.get('ready_to_publish') or item.get('status')=='verified_ready_for_parent_serial_publication'):continue
   receipt_times={}
   for field in ('review','checker','native','capture','execution','registration_pins'):
    if isinstance(item.get(field),str):
     f=pathlib.Path(item[field]); f=f if f.is_absolute() else D/name/f
     try: receipt_times[field]=f.stat().st_mtime_ns
     except OSError: receipt_times[field]=None
   semantic={k:v for k,v in item.items() if k not in ('utc','at','updated_at','heartbeat')}
   candidates.append({'group':name,'entry':semantic,'receipt_mtimes':receipt_times})
 if not candidates:return
 fresh=[];digests=[]
 for candidate in candidates:
  digest=hashlib.sha256(json.dumps(candidate,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  if digest not in entry.get('publish_wake_handled_digests',[]):fresh.append(candidate);digests.append(digest)
 if not fresh:return
 signature=hashlib.sha256(json.dumps(fresh,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 entry.setdefault('publish_wake_pending',[]).append({'signature':signature,'candidate_digests':digests,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pending_receiver_ack'})
 old=entry['job_id'];entry.setdefault('prior_job_ids',[]).append(old);entry.pop('job_id');entry['ready']=True;entry['last_publish_wake_signature']=signature;entry['publish_wake_retry_at']=0
 entry['changed_condition']='New unaccepted coordinator-approved PUBLISH_QUEUE evidence since terminal acceptance receiver; existing authority unchanged'
 parent=read(D/'dispatch_root/SELF_JOB.json')
 if parent:
  r=db.execute('select status,data from jobs where id=?',(parent['id'],)).fetchone()
  if r and r[0] in {'starting','running'}:entry['parent_self_job']=str(D/'dispatch_root/SELF_JOB.json')
  else:entry.pop('parent_self_job',None)
 put(D/'ROOT_CONTROL_QUEUE.json',queue)
 put(D/'acceptance_root/WAKE_REQUEST.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_job':old,'signature':signature,'candidate_digests':digests,'candidates':fresh,'reason':entry['changed_condition'],'status':'pending_receiver_ack','ack_file':str(D/'acceptance_root'/'WAKE_DISPOSITION.json')})
 record['rejected'].append({'control':'acceptance_root','reason':'new publication evidence queued existing authorized receiver'})

def wake_dispatch_for_owned_work(state,db,deadline,record):
 """Restore the dispatch role without requiring an interactive controller turn."""
 if not dispatch_open(deadline):return
 queue=read(D/'ROOT_CONTROL_QUEUE.json',{'tasks':[]})
 entry=next((e for e in queue['tasks'] if e.get('name')=='dispatch_root'),None)
 if not entry or entry.get('enabled') is False or entry.get('phase') in ('paused','stopped','complete') or entry.get('ready') or not entry.get('job_id'):return
 row=db.execute('select status,data from jobs where id=?',(entry['job_id'],)).fetchone()
 if not row:return
 dat=json.loads(row[1])
 if row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp')):return
 now=time.time();finished=dat.get('finished_at') or entry.get('last_recovery_requested_at') or 0
 if now-finished<60:return
 if now-entry.get('last_recovery_requested_at',0)<120:return
 reasons=[]
 for name in state.get('coordinator_groups',[]):
  p=D/name;g=read(p/'GROUP.json',{})
  if g.get('enabled') is False or g.get('phase') in ('paused','stopped','complete'):continue
  jobs=read(p/'CHILD_JOBS.json',{'jobs':[]}).get('jobs',[])
  for j in jobs:
   jr=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone()
   if jr:
    jd=json.loads(jr[1])
    if jr[0] in ACTIVE or live(jd.get('child_pid'),jd.get('child_stamp')):
     reasons.append({'group':name,'reason':'owned execution still active'});break
  directories={j['directory'] for j in jobs}
  if any(e.get('ready') and e.get('approved_by_coordinator') and e.get('enabled') is not False and e.get('phase') not in ('paused','stopped','complete') and e.get('directory') not in directories for e in read(p/'READY_QUEUE.json',{'tasks':[]}).get('tasks',[])):
   reasons.append({'group':name,'reason':'approved undispatched work needs coordinator'})
  handoff=p/'DISPATCH_HANDOFF.json'
  if handoff.exists() and now-handoff.stat().st_mtime<300 and read(handoff,{}).get('never_submitted_candidates_total',0)>0:
   reasons.append({'group':name,'reason':'fresh coordinator report of unsubmitted candidates; reconcile before dispatch'})
 if not reasons:return
 old=entry['job_id'];entry.setdefault('prior_job_ids',[]).append(old);entry.pop('job_id');entry['ready']=True
 entry['last_recovery_requested_at']=now
 entry['changed_condition']='Dispatch coordinator terminated while authorized active or prepared work remains; resume existing ownership, no worker replay'
 put(D/'ROOT_CONTROL_QUEUE.json',queue)
 put(D/'dispatch_root/AUTONOMOUS_RECOVERY.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_job':old,'reasons':reasons,'individual_result_review':False})
 record['rejected'].append({'control':'dispatch_root','reason':'automatic recovery queued for existing authorized work'})

def main():
 with (D/'FLEET_REFILL.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state=read(D/'STATE.json',{});now=time.time();record={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launched':[],'rejected':[],'groups':[]}
  if not state.get('enabled') or state.get('phase') in ('paused','stopped','complete'):return
  deadline=datetime.datetime.fromisoformat(state['deadline_utc']).timestamp()
  if now>=deadline:record['delivery_target_missed']=True;record['seconds_past_delivery_target']=now-deadline
  if not dispatch_open(deadline):record['stopped']='explicit pause or hard stop';put(D/'FLEET_REFILL_LIVE.json',record);return
  db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True)
  wake_dispatch_for_owned_work(state,db,deadline,record)
  wake_acceptance_for_new_publish(state,db,record)
  dispatch_controls(state,db,deadline,record)
  owned=set();passed={tuple(r['key']) for r in read(D/'ACCEPTED.json',{'rows':[]})['rows']}
  groups=[D/n for n in state.get('coordinator_groups',[])]
  for p in groups:passed|=validated_keys(read(p/'PUBLISH_QUEUE.json',{}))
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
   if not dispatch_open(deadline):record['stopped']='disabled or explicit hard-stop margin';break
   group_control=read(p/'GROUP.json',{})
   if group_control.get('enabled') is False or group_control.get('phase') in ('paused','stopped','complete'):
    record['groups'].append({'name':p.name,'dispatch_disabled':True});continue
   with (p/'CHILD_DISPATCH.lock').open('a') as childlock:
    fcntl.flock(childlock,fcntl.LOCK_EX);jobs=read(p/'CHILD_JOBS.json',{'jobs':[]})['jobs'];selfjob=read(group_control.get('supervisor_self_job',p/'SELF_JOB.json'))
    if not selfjob:record['rejected'].append({'group':p.name,'reason':'waiting for queued real supervisor launch'});continue
    parentrow=db.execute('select status,data from jobs where id=?',(selfjob['id'],)).fetchone()
    if not parentrow or parentrow[0] not in {'starting','running'}:
     previous=json.loads(parentrow[1]) if parentrow else {}
     if parentrow and (parentrow[0] in ACTIVE or live(previous.get('child_pid'),previous.get('child_stamp'))):
      record['rejected'].append({'group':p.name,'reason':'old coordinator still stopping/alive; no supervision transfer'});continue
     child_active=False
     for j in jobs:
      row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone()
      if row:
       dat=json.loads(row[1]);child_active|=row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp'))
     dirs={j['directory'] for j in jobs};groupkeys={key(r) for r in read(p/'SCOPE.json',[])}
     actionable=any(e.get('directory') not in dirs and eligibility(e,p,groupkeys,owned,passed,db) is None for e in read(p/'READY_QUEUE.json',{'tasks':[]})['tasks'])
     if child_active or actionable:restore_supervision(p,selfjob,group_control,state,db,deadline,record)
     else:record['groups'].append({'name':p.name,'coordinator_terminal_no_actionable_work':True})
     continue
    active=0
    for j in jobs:
     row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone();d=json.loads(row[1])
     active+=row[0] in ACTIVE or live(d.get('child_pid'),d.get('child_stamp'))
    group=read(p/'GROUP.json');limit=group.get('children',7);groupkeys={key(r) for r in read(p/'SCOPE.json',[])};queue=read(p/'READY_QUEUE.json',{'tasks':[]})['tasks'];record['groups'].append({'name':p.name,'active':active,'limit':limit,'ready_entries':sum(bool(x.get('ready')) and x.get('directory') not in {j['directory'] for j in jobs} for x in queue)})
    for entry in queue:
     if not dispatch_open(deadline):record['stopped']='disabled or explicit hard-stop margin';break
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
     if state.get('require_explicit_host_assignment') and entry.get('allow_npu',True):
      assignment=read(q/'HOST_ASSIGNMENT.json',{})
      prefix='【当前主机分工】设备任务仅允许在 '+assignment['host']+' 执行；设备号以该host guard逻辑卡为准。禁止worker自行选别的空闲host。外部占用时报告协调层，由其核对三机和负载并更新HOST_ASSIGNMENT.json；旧预约撤销/释放、无重叠owner后才迁移。不得因排队超时重建agent。\n'
      task_text=(q/'TASK.txt').read_text()
      if not task_text.startswith(prefix):(q/'TASK.txt').write_text(prefix+task_text)
     cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--write',str(q),'--model','gpt-6-astra','--effort','medium','--timeout',str(native_budget_seconds(3600, allow_unbounded=True)),'--notify','queue','--sandbox','danger-full-access','--parent',tid,'--parent-home',selfjob['home']['name']]
     if entry.get('allow_npu',True):
      # Queue wait is a separate budget from device timeout.  The fleet policy
      # is 600 seconds; entries may request a shorter explicit budget.
      npu_wait=max(1,min(600,int(entry.get('npu_wait',600))))
      cmd+=['--npu-owner',owner,'--npu-timeout',str(min(600,entry.get('npu_timeout',600))),'--npu-wait',str(npu_wait)]
     began=time.time();r=subprocess.run(cmd,capture_output=True,text=True);put(q/'SPAWN.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
     if r.returncode:record['rejected'].append({'group':p.name,'directory':str(q),'reason':'native spawn failed','stderr':r.stderr[-1000:]});break
     native=json.loads(r.stdout);keys={key(r) for r in read(q/'SCOPE.json')};row={'name':p.name+'/'+q.name,'host':'compatible_threehost','count':len(keys),'directory':str(q),'task_file':str(q/'TASK.txt'),'native':native,'submitted_at':began,'returned_at':time.time(),'role':'delegated_integration','coordinator':p.name,'parent_thread':tid,'active_assignment':True};jobs.append(row);put(p/'CHILD_JOBS.json',{'jobs':jobs});owned|=keys;active+=1;record['launched'].append({'name':row['name'],'id':native['id'],'count':len(keys)})
  put(D/'FLEET_REFILL_LIVE.json',record)
  with (D/'FLEET_REFILL_HISTORY.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
  print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
