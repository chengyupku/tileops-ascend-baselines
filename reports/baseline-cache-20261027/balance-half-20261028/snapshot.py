import json, collections, time, sys
from pathlib import Path
sys.path.insert(0, 'tools')
from run_baseline_cache import assigned_history, completed_ids, quarantine_key, identity
host, label = sys.argv[1:]
r = Path('reports/baseline-cache-20261027'); h = r / host
plan = json.loads((r / 'PLAN.json').read_text())
tasks = [t for t in plan['pairs'] if t['host'] == host]
def read(n, default):
 p=h/n
 return json.loads(p.read_text()) if p.exists() else default
def lines(n):
 p=h/n
 return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
latest,cached=assigned_history(tasks, lines('journal.jsonl'), lines('CACHE.jsonl'))
q=read('QUARANTINE.json',{}); retry=read('RETRY.json',{})
done=completed_ids(latest,retry,q,cached)
ready=[t for t in tasks if t['id'] not in done and not quarantine_key(t,q)]
live=read('LIVE.json',{})
ids={}
for k in ['identity','guard_identity']:
 if k in live:
  try: ids[k]=identity(live[k]['pid'])==live[k]
  except OSError: ids[k]=False
summary={'at':time.time(),'host':host,'assigned':len(tasks),'cached':len(cached),'needs_run':len(ready),'ready_by_library':dict(collections.Counter(t['key'][0] for t in ready)), 'live':{k:v for k,v in live.items() if k!='quarantine'}, 'identities_match':ids, 'control_stop':(h/'STOP').read_text() if (h/'STOP').exists() else None}
file=r/'balance-half-20261028'/(host+'-'+label+'.json')
if file.exists(): raise ValueError('snapshot already exists')
file.write_text(json.dumps({**summary,'ready':ready,'cache_ids':list(cached),'done':list(done)},indent=2)+'\n')
print(json.dumps(summary))
