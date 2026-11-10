import collections, fcntl, json, pathlib, sys, time
sys.path.insert(0,'tools')
from run_baseline_cache import assigned_history, completed_ids, quarantine_key, identity
h=sys.argv[1];R=pathlib.Path('reports/baseline-cache-20261027');H=R/h;D=R/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
assert (H/'STOP').read_text()=='inspection-20261029T001544/rebalance-pending-only\n'
with (H/'supervisor.lock').open('a+') as lock:
    until=time.monotonic()+45
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>=until:
                print(json.dumps({'host':h,'waiting_for_normal_pair_end':True,'live':{k:read(H/'LIVE.json').get(k) for k in ['phase','key','pair_elapsed_seconds']}}));sys.exit(75)
            time.sleep(2)
    live=read(H/'LIVE.json');assert live['phase'] in ('sweep_complete','stopped_by_control')
    oldid=live['identity']
    try:alive=identity(oldid['pid'])==oldid
    except OSError:alive=False
    tasks=[t for t in read(R/'PLAN.json')['pairs'] if t['host']==h]
    rows=[json.loads(s) for s in (H/'journal.jsonl').read_text().splitlines() if s.strip()]
    cache=[json.loads(s) for s in (H/'CACHE.jsonl').read_text().splitlines() if s.strip()]
    latest,cached=assigned_history(tasks,rows,cache);q=read(H/'QUARANTINE.json');retry=read(H/'RETRY.json');done=completed_ids(latest,retry,q,cached)
    ready=[t for t in tasks if t['id'] not in done and not quarantine_key(t,q)]
    result={'at':time.time(),'host':h,'supervisor_lock_free':True,'prior_identity_alive':alive,
        'live':{k:live.get(k) for k in ['phase','identity','at','key']},'ready':ready,'cached_ids':sorted(cached),
        'all_cache_ids':sorted({r['task_id'] for r in cache}),
        'status_counts':dict(collections.Counter(r['status'] for r in latest.values())),
        'current_quarantine':q,'retry':retry}
    (D/(h+'-BOUNDARY.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['host','supervisor_lock_free','prior_identity_alive','live']}), 'ready',len(ready),'cached',len(cached))
