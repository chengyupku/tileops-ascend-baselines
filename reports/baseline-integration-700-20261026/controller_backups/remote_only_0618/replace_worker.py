import pathlib,json,subprocess,time,datetime,sys,fcntl
P=pathlib.Path(__file__).resolve().parent

def dispatch_budget(state, group, now=None):
 now=time.time() if now is None else now
 closed=('paused','stopped','complete','completed','dispatch_exhausted','exhausted','hold','api_unsupported')
 if not state.get('enabled') or str(state.get('phase','')).lower() in closed or group.get('enabled') is False or str(group.get('phase','')).lower() in closed:
  raise SystemExit('Dispatch disabled by current task/group state; no native spawn')
 # deadline_utc is an immutable delivery SLA, never an implicit stop.
 budget=0 if state.get('execution_wall_timeout_disabled') else int(group.get('worker_timeout_seconds',state.get('worker_timeout_seconds',3600)))
 limits=[datetime.datetime.fromisoformat(x).timestamp() for x in (state.get('dispatch_stop_at'),group.get('dispatch_stop_at')) if x]
 if limits: budget=min(budget,int(min(limits)-now)) if budget else int(min(limits)-now)
 if budget==0 and not limits:return 0
 if budget<60:raise SystemExit('Explicit dispatch stop or worker budget reached; no native spawn')
 return budget

def ensure_dispatch_enabled():
 return dispatch_budget(json.loads((P.parent/'STATE.json').read_text()),json.loads((P/'GROUP.json').read_text()))

def ensure_released(clearance, previous_id):
 # Terminal CLI status alone does not prove ownership release or device drain.
 if clearance.get('previous_job_id')!=previous_id or not all(clearance.get(k) is True for k in ('legal_unfinished','ownership_released','device_drained')):
  raise SystemExit('Fresh matching legal-work/ownership-release/device-drain clearance required')

name=sys.argv[1];home=sys.argv[2]
with (P/'CHILD_DISPATCH.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 ensure_dispatch_enabled()
 data=json.loads((P/'CHILD_JOBS.json').read_text());r=next(r for r in data['jobs'] if pathlib.Path(r['directory']).name==name);n=r['native']
 z=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True);s=json.loads(z.stdout)
 assert s['status'] in ['failed','completed','cancelled','lost','timed_out'],s['status']
 q=P/name; budget=ensure_dispatch_enabled()
 if json.loads((P.parent/'STATE.json').read_text()).get('require_explicit_host_assignment'):
  assignment=json.loads((q/'HOST_ASSIGNMENT.json').read_text()) if (q/'HOST_ASSIGNMENT.json').exists() else {}
  if assignment.get('host') not in {'local','openlibing-DevEnv-055409','openlibing-DevEnv-661495'} or not assignment.get('approved_by_coordinator'):raise SystemExit('Approved host assignment required before replacement')
  rows=json.loads((q/'SCOPE.json').read_text());scope_keys=sorted([r.get('key',[r.get('baseline'),r.get('operator'),r.get('case_id')]) for r in rows])
  if assignment.get('scope_keys')!=scope_keys:raise SystemExit('Host assignment exact scope mismatch')
  task_path=q/(sys.argv[3] if len(sys.argv)>3 else 'TASK.txt')
  prefix='【当前主机分工】设备任务只能在 '+assignment['host']+' 的host guard执行。禁止复用自主HOSTS轮询选择循环；协调层批准并更新HOST_ASSIGNMENT.json且旧预约释放后才允许迁移。排队不是回收agent理由，使用原进程继续等待或报告协调层，不自行启动同范围替代agent。\n'
  content=task_path.read_text()
  if not content.startswith(prefix):task_path.write_text(prefix+content)
 clearance=q/'REPLACEMENT_CLEARANCE.json'
 ensure_released(json.loads(clearance.read_text()) if clearance.exists() else {},n['id'])
 cmd=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd','/root/dyq/workspace','--task-file',str(q/(sys.argv[3] if len(sys.argv)>3 else 'TASK.txt')),'--write',str(q),'--home',home,'--model','gpt-6-astra','--effort','medium','--timeout',str(budget),'--notify','queue','--sandbox','danger-full-access','--parent',r['parent_thread'],'--parent-home',n['parent_home']['name'],'--npu-owner','baseline700-20261026/'+P.name+'/'+name,'--npu-timeout','600','--npu-wait','600']
 ensure_dispatch_enabled()
 z=subprocess.run(cmd,capture_output=True,text=True);rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_native':n,'previous_status':s,'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr,'condition_changed':sys.argv[4] if len(sys.argv)>4 else 'Capacity failure; switched account home retaining model/medium and exact scope.'}
 with (P/'REPLACEMENTS.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 if z.returncode:print(z.stderr);sys.exit(z.returncode)
 r.setdefault('previous_jobs',[]).append(n);r['native']=json.loads(z.stdout);r['returned_at']=time.time();(P/'CHILD_JOBS.json').write_text(json.dumps(data,indent=2));print(json.dumps({'worker':name,'replacement':r['native']['id'],'home':home}))
