import pathlib,fcntl,ast
D=pathlib.Path(__file__).resolve().parents[1];p=D/'refill_fleet.py'
helper='''def restore_supervision(p,old,group,state,db,deadline,record):
 """Queue a real coordinator only for surviving children or eligible ready work."""
 name='supervise_'+p.name+'_'+old['id'][:8]
 q=D/'coordinator_replacements'/name;q.mkdir(parents=True,exist_ok=True)
 task=q/'TASK.txt'
 if not task.exists():
  task.write_text(f"""你是恢复监督的真实coordinator，gpt-6-astra medium。首先读AGENTS.md、MEMORY索引和codex-workers SKILL，只读必要当前记忆。截止{state['deadline_utc']}不重置。你自己禁止NPU；允许原生manual-quota wrapper + medium + --notify queue派实际执行子worker，每父最多8，所有设备动作前台host npu-guard。\n接管监督目录{p}，旧coordinator已终态，既有CHILD_JOBS中的实际执行子worker所有权保持原样，不能取消或重复派遣，不能伪造parent。refill将GROUP.supervisor_self_job切到你的真实SELF_JOB，所有后续新子任务走READY_QUEUE/refill_fleet，现有活跃children计入组上限。\n只做必要任务分工、CPU准备及补位，禁止写ACCEPTED/providers/memory。用户当前仅控制层汇总，数值验收交acceptance_root真实线程。读BASELINE_WORKER_PROMPT_RULES_20261027.txt。按host兼容与预计分钟均衡，不以代表失败跳过未跑exactkey；无changed condition不重复失败；已完整成功不再跑。原kernel/compiler/pass/schedule/reference/seed/tolerance不改。\n先核现有children PID+启动身份和READY_QUEUE；只在旧owner完成释放、进程设备任务停止后转移范围。CPU准备在guard外，外部占用核另外两host并迁移兼容任务，不杀外部进程或改锁。你获准写报告目录下本组分工/队列/monitor和私有任务；STATE/PLAN/JOBS持DISPATCH.lock，CHILD_JOBS持CHILD_DISPATCH.lock，FLEET变更持FLEET_REFILL.lock。每2分钟HEARTBEAT，每5分钟措施；实在无合法可执行工作则写DISPATCH_HANDOFF，别为凑24制造任务。持续至职责完成/截止。完成通知由native supervisor处理，勿主动queue顶层主会话。\n""",encoding='utf-8')
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

'''
with (D/'FLEET_REFILL.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);s=p.read_text();assert 'def restore_supervision' not in s
 s=s.replace('def main():',helper+'def main():')
 old="    fcntl.flock(childlock,fcntl.LOCK_EX);jobs=read(p/'CHILD_JOBS.json',{'jobs':[]})['jobs'];selfjob=read(group_control.get('supervisor_self_job',p/'SELF_JOB.json'));parentrow=db.execute('select status,data from jobs where id=?',(selfjob['id'],)).fetchone()\n    if not parentrow or parentrow[0] not in {'starting','running'}:record['rejected'].append({'group':p.name,'reason':'coordinator inactive; root must restore supervision'});continue"
 new="""    fcntl.flock(childlock,fcntl.LOCK_EX);jobs=read(p/'CHILD_JOBS.json',{'jobs':[]})['jobs'];selfjob=read(group_control.get('supervisor_self_job',p/'SELF_JOB.json'))
    if not selfjob:record['rejected'].append({'group':p.name,'reason':'waiting for queued real supervisor launch'});continue
    parentrow=db.execute('select status,data from jobs where id=?',(selfjob['id'],)).fetchone()
    if not parentrow or parentrow[0] not in {'starting','running'}:
     child_active=False
     for j in jobs:
      row=db.execute('select status,data from jobs where id=?',(j['native']['id'],)).fetchone()
      if row:
       dat=json.loads(row[1]);child_active|=row[0] in ACTIVE or live(dat.get('child_pid'),dat.get('child_stamp'))
     dirs={j['directory'] for j in jobs};groupkeys={key(r) for r in read(p/'SCOPE.json',[])}
     actionable=any(e.get('directory') not in dirs and eligibility(e,p,groupkeys,owned,passed,db) is None for e in read(p/'READY_QUEUE.json',{'tasks':[]})['tasks'])
     if child_active or actionable:restore_supervision(p,selfjob,group_control,state,db,deadline,record)
     else:record['groups'].append({'name':p.name,'coordinator_terminal_no_actionable_work':True})
     continue"""
 assert old in s;s=s.replace(old,new);ast.parse(s);p.write_text(s)
 print('Added automatic real coordinator replacement only for surviving children or eligible exclusive ready work; no inactive-group restart without work')
