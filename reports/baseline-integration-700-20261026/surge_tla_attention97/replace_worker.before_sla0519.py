import pathlib,json,subprocess,time,datetime,sys,fcntl
P=pathlib.Path(__file__).resolve().parent

def ensure_dispatch_enabled():
 state=json.loads((P.parent/'STATE.json').read_text());group=json.loads((P/'GROUP.json').read_text())
 if not state.get('enabled') or state.get('phase') in ('paused','stopped','complete') or group.get('enabled') is False or group.get('phase') in ('paused','stopped','complete','dispatch_exhausted'):
  raise SystemExit('Dispatch disabled by current task/group state; no native spawn')
 limits=[datetime.datetime.fromisoformat(x).timestamp() for x in (state['deadline_utc'],group['deadline_utc'],'2026-10-27T05:31:00+00:00')]
 if time.time()>=min(limits)-60:raise SystemExit('Original deadline dispatch margin reached; no native spawn')

name=sys.argv[1];home=sys.argv[2]
with (P/'CHILD_DISPATCH.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 ensure_dispatch_enabled()
 data=json.loads((P/'CHILD_JOBS.json').read_text());r=next(r for r in data['jobs'] if pathlib.Path(r['directory']).name==name);n=r['native']
 z=subprocess.run(['python3','/root/dyq/tools/codex-worker.py','status',n['id']],capture_output=True,text=True);s=json.loads(z.stdout)
 assert s['status'] in ['failed','completed','cancelled','lost','timed_out'],s['status']
 q=P/name; deadline=datetime.datetime.fromisoformat(json.loads((P/'GROUP.json').read_text())['deadline_utc']).timestamp()
 assert deadline-time.time()>60,'No dispatch in last minute'
 cmd=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd','/root/dyq/workspace','--task-file',str(q/(sys.argv[3] if len(sys.argv)>3 else 'TASK.txt')),'--write',str(q),'--home',home,'--model','gpt-6-astra','--effort','medium','--timeout',str(max(60,min(3600,int(deadline-time.time())))),'--notify','queue','--sandbox','danger-full-access','--parent',r['parent_thread'],'--parent-home',n['parent_home']['name'],'--npu-owner','baseline700-20261026/'+P.name+'/'+name,'--npu-timeout','600','--npu-wait','600']
 ensure_dispatch_enabled()
 z=subprocess.run(cmd,capture_output=True,text=True);rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'previous_native':n,'previous_status':s,'returncode':z.returncode,'stdout':z.stdout,'stderr':z.stderr,'condition_changed':sys.argv[4] if len(sys.argv)>4 else 'Capacity failure; switched account home retaining model/medium and exact scope.'}
 with (P/'REPLACEMENTS.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 if z.returncode:print(z.stderr);sys.exit(z.returncode)
 r.setdefault('previous_jobs',[]).append(n);r['native']=json.loads(z.stdout);r['returned_at']=time.time();(P/'CHILD_JOBS.json').write_text(json.dumps(data,indent=2));print(json.dumps({'worker':name,'replacement':r['native']['id'],'home':home}))
