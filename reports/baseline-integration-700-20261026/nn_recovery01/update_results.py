import json,datetime,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(f,d):
 q=P/(f+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(P/f)
rows=[];done=set()
for batch in sorted(P.glob('*/GUARD_RESULT.json')):
 if not (batch.parent/'COMMAND.json').exists():continue
 name=batch.parent.name;jobs=json.loads((P/(name+'.json')).read_text());g=json.loads(batch.read_text())
 for base,op,cid,a in jobs:
  d=P/a;c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {};v=c.get('verification',{});ch=json.loads((d/'CHILD_RESULT.json').read_text()) if (d/'CHILD_RESULT.json').exists() else {};m=json.loads((d/'GUARD_METADATA.json').read_text()) if (d/'GUARD_METADATA.json').exists() else {}
  good=g['returncode']==0 and c.get('status')=='verified' and v.get('checked_outputs')==v.get('total_outputs') and v.get('total_outputs',0)>0
  reason='Full canonical checker passed; native attribution requires review before merge.' if good else ('Guard '+str(g['returncode'])+'; no admissible numerical verdict.' if g['returncode'] else str(c.get('error',c.get('status','Not reached'))))
  queue=None
  if m.get('updated_at'):queue=(datetime.datetime.fromisoformat(m['updated_at'])-datetime.datetime.fromisoformat(g['requested_at'])).total_seconds()
  row=dict(baseline=base,operator=op,case_id=cid,status='verified_unregistered' if good else ('failed' if g['returncode']==0 and c.get('status') in ['failed','cleanup_failed'] and c.get('stage')=='verify' else 'blocked'),reason=reason,api=c.get('api'),guard_receipt=str(batch),checker_receipt=str(d/'checker.json') if c else None,checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs'),registered=False,factory='integration700_nn_recovery01.make' if good else None,changed_files=[str(d/'adapter.py')],timings=dict(queue_s=queue,case_s=ch.get('elapsed_s'),guard_total_s=g['elapsed_s'],compile_s=None),evidence=[str(d/'case.log'),str(d/'native.jsonl')],attempt=a)
  rows.append(row);done.add((base,op,cid))
history=rows; latest={}
for r in rows:
 key=(r['baseline'],r['operator'],r['case_id']); stamp=json.loads(Path(r['guard_receipt']).read_text())['requested_at']
 if key not in latest or stamp>latest[key][0]:latest[key]=(stamp,r)
rows=[r for _,r in latest.values()]
scope=json.loads((P/'SCOPE.json').read_text());save('RESULT.json',dict(updated_at=now(),rows=rows,attempt_history=history,remaining_keys=[[r['baseline'],r['operator'],r['case_id']] for r in scope if (r['baseline'],r['operator'],r['case_id']) not in done]))
print({s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)})
