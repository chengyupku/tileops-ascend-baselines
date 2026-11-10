from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());rows=[]
for original in scope:
 key=original['key'];op,cid=key[1:];attempts=[]
 for root in ['local_runs','server1_runs','server2_runs']:
  for d in sorted((P/root).glob(cid+'-*')):
   if not (d/'guard.json').exists():continue
   g=json.loads((d/'guard.json').read_text());c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else None
   m=json.loads((d/'GUARD_RUNNING.json').read_text()) if (d/'GUARD_RUNNING.json').exists() else {}
   code=g.get('returncode');status='guard_not_started' if code==75 else 'device_timeout' if code==124 else 'verified_pending_review' if code==0 and c and c.get('status')=='verified' else 'checker_failed' if c else 'execution_incomplete'
   a=dict(host=root,guard_exit=code,status=status,guard=str(d/'guard.json'),checker=str(d/'checker.json') if c else None,native=str(d/'native.json') if (d/'native.json').exists() else None,guard_log=str(d/'guard.log'),checker_error=str(c.get('error',''))[-1600:] if c else None,seed=c.get('input_seed') if c else None,requested=m.get('requested_at',g.get('requested_utc')),acquired=m.get('acquired_at'),child_start=m.get('updated_at'),end=g.get('end_utc',g.get('finished')),native_child=m.get('child'),timings=dict(wall_seconds=g['finished']-g['started'],checker_wall_seconds=c.get('wall_seconds') if c else None,lock_queue_seconds=None,physical_idle_wait_seconds=None,held_cpu_seconds=None,device_seconds=None))
   if c and c.get('error_type') in ['TVMError','InternalError']:a['status']='native_compile_failed'
   
   if m.get('requested_at') and m.get('acquired_at'):
    a['timings']['lock_queue_seconds']=(datetime.datetime.fromisoformat(m['acquired_at'])-datetime.datetime.fromisoformat(m['requested_at'])).total_seconds()
   if m.get('updated_at') and m.get('acquired_at'):
    a['timings']['acquired_to_child_start_seconds']=(datetime.datetime.fromisoformat(m['updated_at'])-datetime.datetime.fromisoformat(m['acquired_at'])).total_seconds()
   (d/'RESULT.json').write_text(json.dumps(dict(key=key,**a,registered=False,accepted_added=0),indent=2));attempts.append(a)
 preserved=None
 for d in [P/'preserved_topk',P/('preserved_'+op)]:
  f=d/'RESULT.json'
  if f.exists():
   q=json.loads(f.read_text())
   if q['key']==key:preserved=q
 active=[a for a in attempts if a['guard_exit']!=75]
 status=active[-1]['status'] if active else preserved['status'] if preserved else 'not_run' if not attempts else 'guard_not_started'
 row=dict(key=key,baseline=key[0],operator=op,case_id=cid,status=status,accepted_added=0,registered=False,runtime_ready=False,numerical_success=any(a['status']=='verified_pending_review' for a in attempts),numerical_failure_established=False,attempts=attempts,prior_scope_path=original['prior_scope_path'],preserved=preserved,remaining_reason=None if status=='verified_pending_review' else (active[-1]['checker_error'] if active and active[-1]['checker_error'] else preserved['remaining_reason'] if preserved and 'remaining_reason' in preserved else 'No successful complete original checker; no representative substituted'))
 rows.append(row)
(P/'RESULT.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),keys=len(rows),statuses={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))},accepted_added=0),indent=2))
