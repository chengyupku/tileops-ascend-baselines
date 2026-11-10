import pathlib,json,fcntl,datetime,os
D=pathlib.Path(__file__).resolve().parents[1];P=D/'recovery_native5'
def put(p,x):
 t=p.with_name(p.name+'.dispatch.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
with (D/'FLEET_REFILL.lock').open('a') as fleet:
 fcntl.flock(fleet,fcntl.LOCK_EX)
 with (P/'CHILD_DISPATCH.lock').open('a') as child:
  fcntl.flock(child,fcntl.LOCK_EX)
  if json.loads((P/'CHILD_JOBS.json').read_text())['jobs']:
   print('Recovery coordinator already activated and dispatched; idempotent no-op');raise SystemExit(0)
  transfer=json.loads((D/'dispatch_root/TERMINAL_TRANSFER.json').read_text());process=json.loads((P/'PREDECESSOR_PROCESS_CHECK.json').read_text());assert all(x['rc']==0 and not x['matching_processes'] for x in process['hosts'])
  changed=json.loads((P/'CHANGED_CONDITION.json').read_text());assert changed['matches_deployed_summary']
  queue=json.loads((P/'PREPARED_QUEUE.json').read_text())
  for e in queue['tasks']:e.update(ready=True,approved_by_coordinator=True,approved_by='dispatch_root authorized scheduling supervisor',previous_devices_quiescent=True,changed_condition=changed['changed_condition'])
  queue['note']='Activated under authorized dispatch-root terminal-owner handoff; exact prior process identities and threehost empty process checks recorded. No active scope was split.'
  put(P/'OWNERSHIP_TRANSFER.json',transfer);put(P/'READY_QUEUE.json',queue)
  group=json.loads((P/'GROUP.json').read_text());group.update(enabled=True,phase='running');put(P/'GROUP.json',group)
 with (D/'DISPATCH.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state=json.loads((D/'STATE.json').read_text());assert state['enabled'] and datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(state['deadline_utc'])
  if P.name not in state['coordinator_groups']:state['coordinator_groups'].append(P.name)
  put(D/'STATE.json',state)
print('Activated 2 exclusive native execution tasks; historical Bmm3 remains unqueued')
