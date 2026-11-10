import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,'tools')
from run_baseline_cache import assigned_history,completed_ids,quarantine_key
host=sys.argv[1];assert host in ('local','server2')
R=Path('reports/baseline-cache-20261027');H=R/host;D=R/'add-server1-20261028';D.mkdir(exist_ok=True)
token='add-server1-20261028/ownership-boundary\n'
assert not (R/'STOP').exists()
if (H/'STOP').exists():assert (H/'STOP').read_text()==token
else:
    with (H/'STOP').open('x') as f:f.write(token)
plan=json.loads((R/'PLAN.json').read_text());tasks=[t for t in plan['pairs'] if t['host']==host]
rows=[json.loads(x) for x in (H/'journal.jsonl').read_text().splitlines()]
cache=[json.loads(x) for x in (H/'CACHE.jsonl').read_text().splitlines()]
q=json.loads((H/'QUARANTINE.json').read_text());retry=json.loads((H/'RETRY.json').read_text())
latest,cached=assigned_history(tasks,rows,cache);done=completed_ids(latest,retry,q,cached)
l=json.loads((H/'LIVE.json').read_text())
ready=[t for t in tasks if t['id'] not in done and not quarantine_key(t,q)]
s=dict(at=time.time(),host=host,plan_sha256=hashlib.sha256((R/'PLAN.json').read_bytes()).hexdigest(),cached_ids=sorted(cached),ready=ready,
       identity=l['identity'],phase=l['phase'],active_key=l.get('key'),stop=token)
(D/(host+'-boundary.json')).write_text(json.dumps(s,indent=2)+'\n')
print(json.dumps({k:v for k,v in s.items() if k not in ('ready','cached_ids')},ensure_ascii=False));print('ready',len(ready),'cached',len(cached))
