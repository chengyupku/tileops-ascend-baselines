import pathlib,json,os,fcntl,subprocess,time,datetime,sqlite3
P=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/root/dyq/workspace');D=P.parent
with (P/'CHILD_DISPATCH.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 selfjob=json.loads((P/'SELF_JOB.json').read_text());ev=pathlib.Path(selfjob['directory'])/'events.jsonl'
 tid=None
 for line in ev.open():
  e=json.loads(line)
  if e.get('type')=='thread.started':tid=e['thread_id'];break
 assert tid,'Native coordinator thread not initialized yet'
 deadline=datetime.datetime.fromisoformat(json.loads((P/'GROUP.json').read_text())['deadline_utc']).timestamp()
 assert time.time()<deadline
 out=P/'CHILD_JOBS.json';jobs=json.loads(out.read_text())['jobs'] if out.exists() else []
 for q in sorted(P.glob('worker[0-9][0-9]')):
  if any(x['directory']==str(q) for x in jobs):continue
  owner='baseline700-20261026/'+P.name+'/'+q.name
  cmd=['python3',str(W/'.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py'),'--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd',str(W),'--task-file',str(q/'TASK.txt'),'--write',str(q),'--model','gpt-6-astra','--effort','medium','--timeout',str(max(60,min(3600,int(deadline-time.time())))),'--notify','queue','--sandbox','danger-full-access','--parent',tid,'--parent-home',selfjob['home']['name'],'--npu-owner',owner,'--npu-timeout','600','--npu-wait','120']
  began=time.time();r=subprocess.run(cmd,capture_output=True,text=True);receipt={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(q/'SPAWN.json').write_text(json.dumps(receipt,indent=2))
  if r.returncode:raise RuntimeError(r.stderr)
  native=json.loads(r.stdout);row={'name':P.name+'/'+q.name,'host':'compatible_threehost','count':len(json.loads((q/'SCOPE.json').read_text())),'directory':str(q),'task_file':str(q/'TASK.txt'),'native':native,'submitted_at':began,'returned_at':time.time(),'role':'delegated_integration','coordinator':P.name,'parent_thread':tid,'active_assignment':True};jobs.append(row)
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps({'jobs':jobs},indent=2));os.replace(tmp,out)
  print(json.dumps({'name':row['name'],'id':native['id'],'home':native['home']['name'],'count':row['count']}),flush=True)
