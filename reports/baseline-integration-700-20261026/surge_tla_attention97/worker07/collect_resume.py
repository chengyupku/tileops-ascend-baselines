from pathlib import Path
import json,datetime,ast
P=Path(__file__).resolve().parent

def collect():
 rows=json.loads((P/'first_delivery/RESULT.json').read_text());keys=json.loads((P/'REMAINING.json').read_text());attempts=[]
 for pattern in ['local_runs/*/guard.json','server1_evidence/runs/*/guard.json','server2_evidence/runs/*/guard.json','resume_runs/*/guard.json']:
  for f in P.glob(pattern):
   d=json.loads(f.read_text());o=f.parent;m=json.loads((o/'GUARD_RUNNING.json').read_text()) if (o/'GUARD_RUNNING.json').exists() else {}
   c=json.loads((o/'checker.json').read_text()) if (o/'checker.json').exists() else None
   a=dict(key=d['key'],host=d['host'],guard=str(f),argv=d['argv'],requested_at=d['requested_at'],acquired_at=m.get('acquired_at'),child_start=m.get('updated_at'),child=m.get('child'),end=d['ended_at'],guard_exit=d['returncode'],wall_seconds=d['end']-d['start'],checker=str(o/'checker.json') if c else None,checker_status=c.get('status') if c else None,changed_condition=d.get('changed_condition'),held_cpu_seconds=None,device_seconds=None)
   if m.get('acquired_at'):
    a['queue_seconds']=(datetime.datetime.fromisoformat(m['acquired_at'])-datetime.datetime.fromisoformat(d['requested_at'])).total_seconds() if d['host']=='local' else None
    a['queue_note']='remote local-receipt clock offset; use host-native GUARD_RUNNING timestamps' if d['host']!='local' else None
   else:a['queue_seconds']=None
   attempts.append(a)
   r=next(x for x in rows if x['key']==d['key'])
   if d['key'] not in keys:continue
   if c:
    v=c.get('verification',{});ok=c.get('status')=='verified' and d['returncode']==0 and bool(v.get('checked_outputs')) and v.get('checked_outputs')==v.get('total_outputs') and c.get('inputs_preserved') is True
    numeric=c.get('stage')=='verify' and c.get('status')=='failed' and ('Tensor-likes are not close' in c.get('error','') or 'Mismatched elements' in c.get('error',''))
    r.update(status='verified_pending_parent_review' if ok else ('original_compiler_rejected_exact_case' if 'Unresolved call Op' in c.get('error','') else 'exact_numerical_failure' if numeric else 'exact_checker_failed' if c.get('status')=='failed' else 'checker_not_verified'),numerical_success=ok,numerical_failure_established=numeric,checker=str(o/'checker.json'),native=str(o/'native.json') if (o/'native.json').exists() else None,guard=str(f),registration_ready=False,reason=c.get('error') or c.get('reason') or c.get('status'),verification=v,input_seed=c.get('input_seed'))
   elif m:
    r.update(status='device_child_started_no_checker',reason='Child started; inspect exact guard.log for deployment/compiler exception',guard=str(f),checker=None,numerical_failure_established=False)
   elif not r.get('numerical_success') and r.get('checker') is None:
    r.update(status='individually_unverified_resource_blocked',reason='Guard did not start child; not a numerical failure',guard=str(f),numerical_failure_established=False)
 for r in rows:
  own=[a for a in attempts if a['key']==r['key']]
  if own:r['attempts']=own
  if r['key'] in keys:
   (P/(r['key'][1]+'--'+r['key'][2]+'.RESULT.json')).write_text(json.dumps(r,indent=2))
 (P/'RESULT.json').write_text(json.dumps(rows,indent=2))
 summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_type='actual integration / supplemental verification',keys=13,accepted_added=0,verified_added=sum(r.get('numerical_success',False) for r in rows),device_child_started=sum(a['child'] is not None for a in attempts),guard_attempts=len(attempts),guard_wall_seconds=sum(a['wall_seconds'] for a in attempts),status_counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},attempts=attempts,ready=False)
 (P/'DELIVERY.json').write_text(json.dumps(summary,indent=2));return summary
if __name__=='__main__':
 s=collect();print(json.dumps({k:v for k,v in s.items() if k!='attempts'},indent=2))
