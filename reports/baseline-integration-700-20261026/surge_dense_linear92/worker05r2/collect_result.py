from pathlib import Path
import json,datetime,sys
W=Path(__file__).resolve().parent;A=W/sys.argv[1];d=json.loads((A/'checker.json').read_text()) if (A/'checker.json').exists() else {};g=json.loads((A/'GUARD_RESULT.json').read_text());start=json.loads((A/'STARTED.json').read_text());case=start['case'];owner='baseline700-20261026/surge_dense_linear92/worker05r2'
meta=[]
for line in (A/'guard_metadata.jsonl').read_text().splitlines():
 try:
  m=json.loads(json.loads(line)['raw'])
  if m.get('owner')==owner:meta.append(m)
 except Exception:pass
last=meta[-1] if meta else {};dt=lambda s:datetime.datetime.fromisoformat(s)
queue=(dt(last['acquired_at'])-dt(last['requested_at'])).total_seconds() if last.get('acquired_at') and last.get('requested_at') else None
row=dict(key=['ops-transformer','GDNFwdOp',case],baseline='ops-transformer',operator='GDNFwdOp',case_id=case,task_type='补测',status='canonical_checker_failed' if d.get('status')=='failed' else ('verified_pending_parent_review' if d.get('status')=='passed' and g['returncode']==0 else 'not_verified'),registered=False,accepted_added=0,checker=d.get('status'),reason=d.get('error'),input_seed=d.get('input_seed'),guard_exit=g['returncode'],new_device_execution=bool(d),checker_receipt=str(A/'checker.json'),guard_receipt=str(A/'GUARD_RESULT.json'),native_receipt=str(A/'dispatch.jsonl'),full_capture_remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_dense_linear92/worker05r2/'+A.name,command=start['command'],timings=dict(requested=last.get('requested_at'),acquired=last.get('acquired_at'),child_start=None,end=g['utc'],queue_seconds=queue,physical_wait_seconds=None,device_seconds=None,guard_elapsed_seconds=g.get('elapsed_s'),held_cpu_seconds=None),total_outputs=2,complete_checker_outputs_saved=True,original_source_binary_mapping='inherited official package identity; exact source-to-binary build mapping unproven',ready=False)
(A/'RESULT.json').write_text(json.dumps(row,indent=2));print(json.dumps({k:row[k] for k in ['case_id','status','reason','guard_exit']}))
rows=[]
for p in sorted(W.glob('attempt*/RESULT.json')):rows.append(json.loads(p.read_text()))
(W/'RESULT.json').write_text(json.dumps(rows,indent=2));(W/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='case complete',exact_key=row['key'],last_output=str(A/'RESULT.json'),next='next unexecuted scoped case'),indent=2))
