"""Audit completed final source measurements and update only this pilot ledger."""
import datetime
import hashlib
import json
from pathlib import Path
R=Path(__file__).resolve().parent
ROOT=R.parents[3]
scope=json.loads((R.parent/'bitwise4_20261030/baseline-scope.json').read_text())['pairs']
summary=[]
for name,op in [('and','BitwiseAndFwdOp'),('or','BitwiseOrFwdOp'),('xor','BitwiseXorFwdOp'),('not','BitwiseNotFwdOp')]:
 p=R/(name+'-production-final.jsonl');g=R/'requests'/(name+'-production-final')/'guard.json'
 if not p.exists() or not g.exists():continue
 guard=json.loads(g.read_text())
 assert guard['exit_code']==0 and guard['performance_valid'],(name,guard)
 rows=[json.loads(l) for l in p.read_text().splitlines()]
 expected={r['case'] for r in scope if r['op']==op}
 assert len(rows)==len(expected) and {r['case_id'] for r in rows}==expected
 for row in rows:
  assert row['correct'] and row['input_unchanged'] and len(row['latency_samples_ms'])==15
  assert len(row['eager']['latency_samples_ms'])==15
  assert row['runtime_limit']=={'cube_core_num':32,'vector_core_num':64}
  for path,digest in row['source_manifest'].items():
   assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 ratios=[r['eager']['latency_ms']/r['latency_ms'] for r in rows]
 summary.append({'op':op,'cases':len(rows),'correct':len(rows),'performance_passes':sum(r>=.8 for r in ratios),'minimum_eager_ratio':min(ratios),'comparison':'same-host same-process-limit eager, full callable msprof 3/15 and 1GiB flush','full_registered_baseline_acceptance':False,'evidence':str(p.relative_to(ROOT))})
(R/'PRODUCTION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
p=R.parent/'elementwise_serial_20261030/STATE.json';state=json.loads(p.read_text())
for item in summary:
 for row in state['operators']:
  if row['op']==item['op']:
   row.update(local32_validated=True,status='baseline_pending' if item['performance_passes']==item['cases'] else 'performance_deferred',detail=f"本地32AIC：正确{item['correct']}/{item['cases']}，同机eager达80% {item['performance_passes']}/{item['cases']}；登记baseline待补齐",evidence=item['evidence'])
state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
t=p.with_suffix('.tmp');t.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');t.replace(p)
print(json.dumps(summary,indent=2))
