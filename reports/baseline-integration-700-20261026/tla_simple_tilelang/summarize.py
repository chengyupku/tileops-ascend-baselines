from pathlib import Path
import json,datetime
P=Path(__file__).resolve().parent
def save(n,v):
 t=P/(n+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(P/n)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
reg=json.loads((P/'REGISTERED.json').read_text()) if (P/'REGISTERED.json').exists() else {'keys':[]}
regs=[reg]+[json.loads(f.read_text()) for f in P.glob('REGISTERED_*.json')]
rows=[]
for f in (P/'runs').glob('*/guard.json'):
 g=json.loads(f.read_text());key=g['key'];c=json.loads((f.parent/'checker.json').read_text()) if (f.parent/'checker.json').exists() else {}
 passed=c.get('status')=='verified' and g['returncode']==0 and c.get('verification',{}).get('checked_outputs',0)>0 and c['verification']['checked_outputs']==c['verification']['total_outputs']
 reg=next((x for x in regs if key[1:] in x['keys']),{'keys':[]});registered=key[1:] in reg['keys']
 v=c.get('verification',{})
 r=dict(zip(['baseline','operator','case_id'],key));r.update(status=('completed' if registered else 'verified_unregistered') if passed else 'failed',reason=c.get('error') or c.get('reason') or g.get('reason') or (None if passed else 'No complete verified checker; guard returncode='+str(g['returncode'])),api=c.get('api'),guard_receipt=str(f),checker_receipt=str(f.parent/'checker.json') if c else None,checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs'),registered=registered if passed else False,factory=reg.get('factory') if registered else None,changed_files=list(reg.get('pins',{})) if registered else list(c.get('source_sha256',{})),timings=dict(queue_seconds=None,guard_wall_seconds=g['finished']-g['started'],checker_wall_seconds=c.get('wall_seconds'),compile_seconds=None),evidence=c.get('source_sha256'))
 meta_path=f.parent/'GUARD_RUNNING.json'
 if meta_path.exists():
  meta=json.loads(meta_path.read_text())
  requested=datetime.datetime.fromisoformat(meta['requested_at'])
  acquired=datetime.datetime.fromisoformat(meta['acquired_at'])
  launched=datetime.datetime.fromisoformat(meta['updated_at'])
  r['timings']['queue_seconds']=(acquired-requested).total_seconds()
  r['timings']['prelaunch_idle_seconds']=(launched-acquired).total_seconds()
  r['timings']['shared_batch_guard']=g.get('shared_guard',False)
 r['last_attempt_finished']=g['finished']
 rows.append(r)
latest={}
for row in rows:
 key=tuple(row[k] for k in ['baseline','operator','case_id'])
 old=latest.get(key)
 if old is None or row['status']=='completed' or (old['status']!='completed' and row['last_attempt_finished']>old['last_attempt_finished']): latest[key]=row
for f in [P/'STATIC_BLOCKERS.json',P/'PRIOR_BLOCKERS.json']:
 if f.exists():
  for row in json.loads(f.read_text())['rows']:
   key=tuple(row[k] for k in ['baseline','operator','case_id'])
   if key in latest:continue
   row.update(guard_receipt=row.get('guard_receipt'),checker_receipt=row.get('checker_receipt'),checked_outputs=0,total_outputs=2 if row['operator'] in {'FP8QuantFwdOp','CausalConv1dUpdateFwdOp','CausalConv1dFwdOp'} else row.get('total_outputs'),registered=False,factory=None,changed_files=[],timings={'queue_seconds':None,'device_seconds':None,'compile_seconds':None})
   latest[key]=row
save('RESULT.json',dict(rows=list(latest.values()),attempts=rows,utc=now,scope_count=80))
save('HEARTBEAT.json',dict(utc=now,phase='select_batch',current_keys=[],last_progress_at=now,blocked_reason=None))
print([(r['operator'],r['case_id'],r['status']) for r in rows])
