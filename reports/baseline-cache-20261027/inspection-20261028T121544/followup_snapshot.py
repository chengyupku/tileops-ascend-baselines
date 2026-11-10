import json, sys, time, collections
from pathlib import Path
sys.path.insert(0, 'tools')
from run_baseline_cache import assigned_history, completed_ids, quarantine_key, identity
host=sys.argv[1]
r=Path('reports/baseline-cache-20261027'); h=r/host; report=r/'inspection-20261028T121544'
def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
plan=read(r/'PLAN.json'); tasks=[t for t in plan['pairs'] if t['host']==host]
rows=lines(h/'journal.jsonl'); caches=lines(h/'CACHE.jsonl'); q=read(h/'QUARANTINE.json'); live=read(h/'LIVE.json')
latest,cached=assigned_history(tasks, rows, caches); done=completed_ids(latest,read(h/'RETRY.json'),q,cached)
ready=[t['id'] for t in tasks if t['id'] not in done and not quarantine_key(t,q)]
held=[t['id'] for t in tasks if t['id'] not in cached and quarantine_key(t,q)]
actions=lines(report/(host+'-hold-actions.jsonl')); added={i for a in actions for i in a['added']}
after=[x for x in rows if x.get('at',0)>actions[0]['at']]
native={}; guard=read(Path('/run/lock/ascend-npu/device-0.lock'))
for name,expected in [('supervisor',live.get('identity')),('guard',live.get('guard_identity')),('foreground_child',guard.get('child'))]:
    if not expected: continue
    try:
        actual=identity(expected['pid']); fields=Path(f'/proc/{expected["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        native[name]=dict(expected=expected,actual=actual,state=fields[0],matches=all(actual[k]==v for k,v in expected.items() if k in actual))
    except OSError as e: native[name]=dict(expected=expected,matches=False,error=str(e))
owned={t['id'] for t in tasks};now=time.time()
out=dict(at=now,host=host,assigned=len(tasks),cached=len(cached),ready=len(ready),held=len(held),
         tested_noncache_unheld=len(tasks)-len(cached)-len(ready)-len(held),
         cached_last_1h=len({x['id'] for x in rows if x['id'] in owned and x['status']=='cached' and x.get('at',0)>=now-3600}),
         cached_last_4h=len({x['id'] for x in rows if x['id'] in owned and x['status']=='cached' and x.get('at',0)>=now-14400}),
         native=native,live={k:live.get(k) for k in ['at','phase','key','pair_elapsed_seconds','pair_phase']},
         held_added=len(added),held_on_disk=sum('pair:'+i in q for i in added),held_in_live=sum('pair:'+i in live.get('quarantine',{}) for i in added),
         new_cached_since_action=len({x['id'] for x in after if x['status']=='cached'}),
         consumed_exact_skips=[x['id'] for x in after if x['id'] in added and x['status']=='temporarily_skipped_library'],
         guard_phase=guard.get('phase'),guard_owner=guard.get('owner'),stop=(h/'STOP').exists())
path=report/(host+'-followup-'+str(time.time_ns())+'.json');path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(report=str(path),**out),ensure_ascii=False))
