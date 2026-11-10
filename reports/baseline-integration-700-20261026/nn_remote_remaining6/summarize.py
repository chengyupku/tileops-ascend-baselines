import json,hashlib,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());results=[]
for i,x in enumerate(scope):
 attempts=[]
 for d in sorted(P.glob(f'case{i}*')):
  if not d.is_dir():continue
  c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
  g=json.loads((d/'GUARD_RESULT.json').read_text()) if (d/'GUARD_RESULT.json').exists() else {}
  a=dict(attempt=d.name,checker_receipt=str(d/'checker.json') if c else None,guard_receipt=str(d/'GUARD_RESULT.json') if g else None,guard_returncode=g.get('returncode'),status=c.get('status','pending'),api=c.get('api'),error=c.get('error'),verification=c.get('verification'),source_sha256=c.get('source_sha256',{}),elapsed_s=g.get('elapsed_s'),checker_wall_s=c.get('wall_seconds'),provider_source_changed=c.get('provider_source_changed'),oracle_source_changed=c.get('oracle_source_changed'),loaded_maps=str(d/'LOADED_MAPS.txt'),loaded_modules=str(d/'LOADED_MODULES.json'),native=str(d/'native.jsonl'),tested_binding=str(d/'adapter.py'),binding_sha256=hashlib.sha256((d/'adapter.py').read_bytes()).hexdigest())
  attempts.append(a)
 good=[a for a in attempts if a['status']=='verified' and a['guard_returncode']==0 and a['verification']['checked_outputs']==a['verification']['total_outputs']]
 results.append({**{k:x[k] for k in ['baseline','operator','case_id']},'status':'verified_unregistered' if good else ('failed' if any(a['status'] in ('failed','cleanup_failed') for a in attempts) else 'pending'),'registered':False,'recommend_acceptance':False,'attempts':attempts,'selected_attempt':good[0]['attempt'] if good else None})
(P/'RESULT.json').write_text(json.dumps(results,indent=2));(P/'TIMINGS.json').write_text(json.dumps([dict(key=[r[k] for k in ['baseline','operator','case_id']],attempts=[{k:a[k] for k in ['attempt','elapsed_s','checker_wall_s','guard_returncode']} for a in r['attempts']]) for r in results],indent=2))
print([(r['baseline'],r['operator'],r['case_id'],r['status']) for r in results])
