from pathlib import Path
import json,datetime
R=Path(__file__).resolve().parent
rows=[]
for attempt in sorted((R/'runs').glob('*')):
 p=attempt/'output/checker.json'
 if not p.exists():continue
 c=json.loads(p.read_text()); g=json.loads((attempt/'GUARD_RESULT.json').read_text()) if (attempt/'GUARD_RESULT.json').exists() else {}
 print(attempt.name,c.get('status'),c.get('api'),c.get('verification'),str(c.get('error',c.get('reason')))[:200])
 if not g:continue
 enclosing='batch03' if attempt.name in ['attempt03','attempt04','attempt05','attempt06','attempt07'] else 'batch05' if 13<=int(attempt.name[-2:])<=17 else None
 if enclosing:
  ep=R/enclosing/'GUARD_RESULT.json'
  if not ep.exists() or json.loads(ep.read_text())['returncode']!=0:continue
 if attempt.name in ['attempt08','attempt09','attempt10','attempt11','attempt12']:continue
 scope=next(x for x in json.loads((R/'SCOPE.json').read_text()) if x['case_id']==c['case_id'] and x['operator']==c['op'])
 v=c.get('verification',{}); passed=c.get('status')=='verified' and g['returncode']==0
 if passed:
  assert v['checked_outputs']==v['total_outputs'] and v['checked_outputs']>0
  assert any(k.endswith('/integration700_sgl_conv_sampling02.py') for k in c['source_sha256'])
 m=json.loads((attempt/'output/GUARD_RUNNING.json').read_text())
 dt=lambda s:datetime.datetime.fromisoformat(s)
 queue=(dt(m['updated_at'])-dt(m['requested_at'])).total_seconds()
 row={k:scope[k] for k in ['baseline','operator','case_id']}
 row.update(status='verified_unregistered' if passed else 'failed',reason='Complete unchanged checker passed; private-only write authorization prevents registration' if passed else c.get('error',c.get('reason','Checker did not verify')),api=c.get('api'),guard_receipt=str(attempt/'GUARD_RESULT.json'),checker_receipt=str(p),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs'),registered=False,factory='integration700_sgl_conv_sampling02.'+('conv' if scope['operator']=='Conv2dFwdOp' else 'chain'),changed_files=[str(R/'integration700_sgl_conv_sampling02.py')],timings={'queue_seconds':None if 'nested_guard' in g else queue,'device_seconds':g['elapsed_seconds'] if 'nested_guard' in g else g['elapsed_seconds']-queue,'compile_seconds':None})
 row['enclosing_guard_receipt']=str(R/enclosing/'GUARD_RESULT.json') if enclosing else row['guard_receipt']
 rows=[x for x in rows if (x['operator'],x['case_id'])!=(row['operator'],row['case_id'])];rows.append(row)
t=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=R/'RESULT.json'; tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'updated_at':t,'rows':rows},indent=2));tmp.replace(p)
p=R/'HEARTBEAT.json';p.write_text(json.dumps(dict(utc=t,phase='receipts_updated',current_keys=[],last_progress_at=t,blocked_reason=None)))
