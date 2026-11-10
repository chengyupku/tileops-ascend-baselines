"""CPU-only promotion after exact pilot's full guarded cache success."""
import json,os,time
from pathlib import Path
O=Path(__file__).resolve().parents[1]/'reports/baseline-old1076-20261029'
record={'pid':os.getpid(),'started_at':time.time(),'phase':'waiting_pilot'}
def save():
 p=O/'RESET_ROLLOUT.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');os.replace(t,p)
pilot=json.loads((O/'RESET_PILOT.json').read_text())['pilot_id'];save()
while time.time()-record['started_at']<3600:
 rows=[json.loads(x) for x in (O/'server1/journal.jsonl').read_text().splitlines()]
 match=[r for r in rows if r['id']==pilot and r.get('repair_revision')=='support-old1076-transformer-reset-v2']
 if match:break
 if (O/'STOP').exists() or (O/'server1/STOP').exists():record['phase']='stopped';save();raise SystemExit(0)
 time.sleep(5)
else:record['phase']='pilot_not_finished_within_hour';save();raise SystemExit(1)
r=match[-1];c=Path(r['measurement']).parent/'correctness.json'
if r['status']!='cached' or not c.exists() or json.loads(c.read_text()).get('status')!='verified':
 record.update(phase='pilot_needs_review',result=r);save();raise SystemExit(0)
p=O/'TRANSFORMER_REMAINING_QUEUE.json';q=json.loads(p.read_text());live=json.loads((O/'server1/LIVE.json').read_text());seen={r['id'] for r in rows};updated=[]
for tid,e in q['pairs'].items():
 if tid in seen or e['key']==live.get('key') or e['entrypoint']!='tools/baseline_old1076_transformer.py':continue
 e.update(entrypoint='tools/baseline_old1076_transformer_reset_v2.py',revision='support-old1076-transformer-reset-v2',reason='Pilot verified and cached: original cleanup at preexisting reset boundaries frees prior workspace; native API binding reused; unchanged checker and profiling protocol')
 updated.append(tid)
t=p.with_suffix('.reset.tmp');t.write_text(json.dumps(q,indent=2)+'\n');os.replace(t,p)
record.update(phase='promoted',pilot=pilot,updated=updated,finished_at=time.time());save()
