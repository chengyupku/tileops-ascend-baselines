from pathlib import Path
import json,datetime,collections,hashlib
P=Path(__file__).resolve().parent
s=json.loads((P/'RESULT.json').read_text());summary=[];timings=[]
def dt(x):
 try:return datetime.datetime.fromisoformat(x).timestamp()
 except:return None
for row in s['exactkeys']:
 r=dict(key=row['key'],status=row['status'],numerical_success=False,numerical_failure_established=False,accepted_added=0,runtime_ready=False,prior_result_path=row['prior_result_path'],prior_scope_path=row['prior_scope_path'],new_attempts=row.get('new_attempts',[]))
 for a in r['new_attempts']:
  f=Path(a['guard']);g=json.loads(f.read_text());m=g.get('metadata',{});run=f.parent;running=json.loads((run/'GUARD_RUNNING.json').read_text()) if (run/'GUARD_RUNNING.json').exists() else {}
  req=dt(m.get('requested_at'));acq=dt(m.get('acquired_at'));child=dt(running.get('updated_at'))
  timings.append(dict(key=row['key'],host=a['host'],guard_exit=g.get('returncode'),requested_at=m.get('requested_at'),acquired_at=m.get('acquired_at'),child_start_at=running.get('updated_at'),end_epoch=g.get('end'),lock_queue_seconds=acq-req if req is not None and acq is not None else None,physical_wait_seconds=child-acq if child is not None and acq is not None else None,guarded_command_and_cleanup_seconds=g['end']-child if child is not None else None,device_execution_seconds=None,held_cpu_seconds=None,held_before_no_launch_seconds=g['end']-acq if acq is not None and child is None else None,total_seconds=g['end']-g['start'],guard=str(f)))
 if isinstance(row.get('checker'),dict):
  c=row['checker'];r.update(checker=str(Path(row['evidence'])/'checker.json'),input_seed=c.get('input_seed'),stage=c.get('stage'),input_snapshot=c.get('case_snapshot'),input_preservation_verified=False,checker_status=c.get('status'),error_type=c.get('error_type'),error_tail=c.get('error','').splitlines()[-1:],observed_calls=c.get('observed_calls'),numerical_success=False,guard_conflict_detected=row['guard'].get('metadata',{}).get('conflict_detected'),guard_exit=row['guard'].get('returncode'))
  if 'Unresolved call Op(tl.ascend_fill)' in c.get('error',''):r['status']='blocked_original_codegen_ascend_fill_exact_attempt' if not r['guard_conflict_detected'] else 'codegen_error_observed_guard_conflicted_not_accepted'
  elif c.get('status')=='passed':r['status']='checker_pass_requires_parent_full_evidence_review'
  else:r['status']='original_checker_error_exact_attempt'
 elif row['status'].startswith('reused'):
  r['prior_exact_result']=row['prior_exact_result'];r['status']='prior_exact_blocker_retained_no_replay'
 elif (row.get('guard') or {}).get('returncode')==78:r['status']='not_started_guard78'
 summary.append(r)
counts=dict(collections.Counter(r['status'] for r in summary))
(P/'REVIEW.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=14,status_counts=counts,accepted_added=0,numerical_successes=0,exactkeys=summary),indent=2))
(P/'TIMINGS.json').write_text(json.dumps(timings,indent=2))
print(json.dumps(counts))
