import json,subprocess,sys,datetime
from pathlib import Path
W=Path(__file__).resolve().parent
scope=json.loads((W/'SCOPE.json').read_text());est=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text());first='q35-397b-tp1-32768-bfloat16'
assert json.loads((W/first/'checker.json').read_text())['status']=='verified'
order=sorted([s['case_id'] for s in scope if s['case_id']!=first],key=lambda c:est[c]['minimum_effective_available_host_bytes'])
(W/'EXECUTION_ORDER.json').write_text(json.dumps([first]+order,indent=2))
def result(stop=None):
 rows=[]
 for s in scope:
  c=s['case_id'];d=W/c;x=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {};g=json.loads((d/'GUARD_RECEIPT.json').read_text()) if (d/'GUARD_RECEIPT.json').exists() else {}
  status=x.get('status') or ('guard_failed' if g else 'capacity_refused' if (d/'PREQUEUE_REFUSAL.json').exists() else 'not_run_after_stop' if stop else 'pending')
  ok=status=='verified' and g.get('returncode')==0 and x.get('inputs_preserved') and x.get('verification',{}).get('checked_outputs')==2 and x['verification']['total_outputs']==2 and x.get('provider_source_changed') is False and x.get('oracle_source_changed') is False
  rows.append({'exactkey':[s['baseline'],s['operator'],c],'status':'verified_pending_parent_review' if ok else status,'registered':False,'accepted_added':0,'checked_outputs':x.get('verification',{}).get('checked_outputs',0),'total_outputs':2,'inputs_preserved':x.get('inputs_preserved'),'input_seed':x.get('input_seed'),'guard_returncode':g.get('returncode'),'queue_seconds':g.get('queue_seconds'),'child_seconds':g.get('child_seconds'),'checker_status':x.get('status'),'stop_reason':stop if not ok else None,'evidence':[str(p.relative_to(W)) for p in d.iterdir() if p.is_file()]})
 (W/'RESULT.json').write_text(json.dumps(rows,indent=2))
result()
stop=None
for c in order:
 p=subprocess.run([sys.executable,'-S','-B',str(W/'run_private.py'),c])
 d=W/c
 g=json.loads((d/'GUARD_RECEIPT.json').read_text()) if (d/'GUARD_RECEIPT.json').exists() else {}
 x=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
 if p.returncode or g.get('returncode')!=0 or x.get('status')!='verified':
  stop={'case':c,'runner_rc':p.returncode,'guard_rc':g.get('returncode'),'checker_status':x.get('status'),'error':x.get('error'),'policy':'Stop same-path batch after first new failure; no retry'}
  result(stop);break
 result()
(W/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'batch_stopped' if stop else 'batch_complete','current_case':c,'stop':stop,'accepted_added':0},indent=2))
print(json.dumps({'complete':not bool(stop),'stop':stop}),flush=True)
