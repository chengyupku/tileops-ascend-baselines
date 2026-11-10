import importlib.util,tempfile,pathlib,json,sqlite3,time
file=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/refill_fleet.py')
spec=importlib.util.spec_from_file_location('fleet',file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cases=0
for mode,expect in [('supervised_active',False),('supervised_terminal',True),('supervised_stale_status_alive',False),('terminal_active',True),('terminal_ready',True),('terminal_empty',False),('live_root',False),('stale_status_alive_pid',False),('paused',False),('deadline',False),('delivery_sla_overdue',True),('disabled_group',False),('failed_cooldown',False)]:
 with tempfile.TemporaryDirectory() as t:
  m.D=pathlib.Path(t);(m.D/'g').mkdir();(m.D/'dispatch_root').mkdir();now=time.time();state={'enabled':mode!='paused','phase':'active','coordinator_groups':['g']}
  
  if mode=='deadline':state['dispatch_stop_at']=__import__('datetime').datetime.fromtimestamp(now-1,__import__('datetime').timezone.utc).isoformat()
  m.put(m.D/'STATE.json',state);m.put(m.D/'ROOT_CONTROL_QUEUE.json',{'tasks':[{'name':'dispatch_root','role':'coordinator','job_id':'old','ready':False}]})
  m.put(m.D/'g/GROUP.json',{'enabled':mode!='disabled_group'})
  jobs=[] if mode in ['terminal_ready','terminal_empty'] else [{'native':{'id':'child'},'directory':str(m.D/'g/child')}]
  m.put(m.D/'g/CHILD_JOBS.json',{'jobs':jobs});m.put(m.D/'g/READY_QUEUE.json',{'tasks':[{'directory':str(m.D/'g/new'),'ready':True,'approved_by_coordinator':True}] if mode=='terminal_ready' else []})
  db=sqlite3.connect(':memory:');db.execute('create table jobs(id text,status text,data text)')
  rootstatus='running' if mode=='live_root' else 'failed'
  db.execute('insert into jobs values(?,?,?)',('old',rootstatus,json.dumps({'finished_at':now-(10 if mode=='failed_cooldown' else 200),'child_pid':123,'child_stamp':'x'})))
  db.execute('insert into jobs values(?,?,?)',('child','running',json.dumps({'child_pid':456,'child_stamp':'y'})))
  if mode.startswith('supervised_'):
   m.put(m.D/'g/SELF_JOB.json',{'id':'group_root'})
   db.execute('insert into jobs values(?,?,?)',('group_root','running' if mode=='supervised_active' else 'completed',json.dumps({'child_pid':789,'child_stamp':'z'})))
  m.live=lambda pid,stamp:(mode=='stale_status_alive_pid' and pid==123) or (mode=='supervised_stale_status_alive' and pid==789)
  record={'rejected':[]};m.wake_dispatch_for_owned_work(state,db,now+(-1 if mode in ['deadline','delivery_sla_overdue'] else 600),record)
  q=m.read(m.D/'ROOT_CONTROL_QUEUE.json')['tasks'][0]
  assert q['ready']==expect,(mode,q)
  if expect:
   assert q['prior_job_ids']==['old'] and 'job_id' not in q
   m.wake_dispatch_for_owned_work(state,db,now+600,record)
   assert m.read(m.D/'ROOT_CONTROL_QUEUE.json')['tasks'][0]['prior_job_ids']==['old']
  cases+=1
print(f'PASS {cases} dispatch recovery scenarios; real PID guard, pause, deadline, cooldown, empty scope, idempotency. No native spawn/NPU calls.')
