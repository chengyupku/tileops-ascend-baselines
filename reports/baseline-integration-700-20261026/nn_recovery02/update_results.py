import json,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
rows=json.loads((P/'SCOPE.json').read_text())
for r in rows:
 r.update(status='blocked',reason='Awaiting this worker short batch; prior attempt was not admissible. Not a numerical failure.',registered=False,checker_receipt=None,guard_receipt=None,checked_outputs=0,total_outputs=None,factory=None,changed_files=[],timings={'queue_s':None,'device_s':None,'compile_s':None})
for jobfile in [P/'fp32_probe01.json',*sorted(P.glob('matrix_batch*.json')),*sorted(P.glob('matrix_remainder*.json')),P/'other01.json']:
 if not jobfile.exists():continue
 guard=P/jobfile.stem/'GUARD_RESULT.json'
 if not guard.exists():continue
 g=json.loads(guard.read_text())
 for base,op,cid,a in json.loads(jobfile.read_text()):
  r=next(x for x in rows if (x['baseline'],x['operator'],x['case_id'])==(base,op,cid));out=P/a;cp=out/'checker.json';c=json.loads(cp.read_text()) if cp.exists() else {};ch=json.loads((out/'CHILD_RESULT.json').read_text()) if (out/'CHILD_RESULT.json').exists() else {}
  valid=g['returncode']==0 and ch.get('returncode')==0 and c.get('status')=='verified' and c.get('verification',{}).get('checked_outputs')==c.get('verification',{}).get('total_outputs') and c.get('verification',{}).get('total_outputs') is not None
  r.update(status='verified_unregistered' if valid else ('blocked' if g['returncode'] in (75,76,78,124) or not c or c.get('status')=='source_changed' else 'failed'),reason='Full canonical checker passed; shared provider writes forbidden in this scope. Frozen registration packet required.' if valid else str(c.get('error') or f"Guard rc={g['returncode']}; checker status={c.get('status')}; not a valid full pass"),api=c.get('api'),guard_receipt=str(guard),checker_receipt=str(cp) if cp.exists() else None,checked_outputs=c.get('verification',{}).get('checked_outputs',0),total_outputs=c.get('verification',{}).get('total_outputs'),factory=str(out/'adapter.py')+':make',changed_files=[str(out/'adapter.py')],evidence=[str(out/'case.log'),str(out/'native.jsonl'),str(out/'DISPATCH.json')],timings={'guard_total_s':g['elapsed_s'],'case_s':ch.get('elapsed_s'),'compile_s':None,'queue_s':None,'device_s':None})
  mp=out/'GUARD_METADATA.json'
  if mp.exists():
   m=json.loads(mp.read_text());r['guard_metadata']=str(mp)
   dt=datetime.datetime.fromisoformat
   r['timings']['queue_s']=(dt(m['acquired_at'])-dt(m['requested_at'])).total_seconds()
   r['timings']['held_s']=(dt(g['finished_at'])-dt(m['acquired_at'])).total_seconds()
   r['timings']['load_s']=(dt(g['finished_at'])-dt(m['updated_at'])).total_seconds()
  if c.get('status')=='source_changed':
   a0=c.get('provider_source_sha256',{});a1=c.get('provider_source_sha256_after',{});r['source_changes']=[k for k in a0.keys()|a1.keys() if a0.get(k)!=a1.get(k)]
   r['reason']='Canonical numerical checks passed but actual checker rejected concurrent provider source changes. No success claimed; no identical numerical rerun. Parent can audit unrelated registrations. Changed: '+str(r['source_changes'])
obj=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows)
tmp=P/'RESULT.json.tmp';tmp.write_text(json.dumps(obj,indent=2));tmp.replace(P/'RESULT.json')
print({s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)})
