"""CPU-only exact-pair evidence ledger; no device execution or inferred failures."""
import collections
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
b=ROOT/'reports/baseline-cache-20261027'
r=b/'support-run-20261029'
host=sys.argv[1]
plan=json.load(open(b/'PLAN.json'))
scope={x['id'] for x in json.load(open(r/'SCOPE.json'))}
owned={x['id']:x for x in plan['pairs'] if x['host']==host and x['id'] in scope}
journal=[json.loads(s) for s in (b/host/'journal.jsonl').read_text().splitlines()]
latest={x['id']:x for x in journal if x['id'] in owned}
caches={json.loads(s)['task_id']:json.loads(s) for s in (b/host/'CACHE.jsonl').read_text().splitlines()}
supported=[]; failures=[]
for ident,task in owned.items():
    row=latest.get(ident,{})
    c_row=caches.get(ident)
    measure=Path(row['measurement']) if row.get('measurement') else None
    m=json.load(open(measure)) if measure and measure.is_file() else {}
    cp=Path((c_row or m).get('correctness_receipt',str(measure.parent/'correctness.json') if measure else '/nonexistent'))
    c=json.load(open(cp)) if cp.is_file() else {}
    v=c.get('verification',{})
    verified=(c.get('status')=='verified' and c.get('inputs_preserved') is True
              and v.get('checked_outputs',0)>0 and v.get('checked_outputs')==v.get('total_outputs')
              and not c.get('provider_source_changed') and not c.get('oracle_source_changed'))
    if verified:
        supported.append({'id':ident,'key':task['key'],'host':host,'bucket':'baseline支持',
                          'correctness_receipt':str(cp),'performance_cached':c_row is not None,
                          'measurement':str(measure) if measure else None,
                          'evidence':'original full checker, all outputs, preserved inputs; timing acceptance separately recorded'})
    else:
        failures.append({'id':ident,'key':task['key'],'host':host,'journal_status':row.get('status'),
                         'measurement':str(measure) if measure else None,'checker_status':c.get('status'),
                         'error':c.get('error',c.get('reason',m.get('error',row.get('error')))),
                         'repair_revision':row.get('repair_revision')})
result={'at':time.time(),'host':host,'assignment_revision':plan['assignment_revision'],
        'owned':len(owned),'supported':supported,'unresolved':failures,
        'cached':sum(x['performance_cached'] for x in supported)}
p=r/(host+'-DECISIONS.json');tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
print(json.dumps({'host':host,'supported':len(supported),'cached':result['cached'],'unresolved':len(failures)}))
