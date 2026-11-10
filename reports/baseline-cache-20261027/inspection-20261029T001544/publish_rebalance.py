import fcntl, hashlib, json, os, pathlib, sys, time
sys.path.insert(0,'tools')
from run_baseline_cache import quarantine_key, identity
h=sys.argv[1];R=pathlib.Path('reports/baseline-cache-20261027');H=R/h;D=R/'inspection-20261029T001544'
read=lambda p:json.loads(p.read_text())
assert (H/'STOP').read_text()=='inspection-20261029T001544/rebalance-pending-only\n'
with (R/'PLAN.lock').open('a+') as plock, (H/'supervisor.lock').open('a+') as lock:
    fcntl.flock(plock,fcntl.LOCK_EX)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    live=read(H/'LIVE.json');assert live['phase'] in ('sweep_complete','stopped_by_control')
    try:alive=identity(live['identity']['pid'])==live['identity']
    except OSError:alive=False
    assert not alive
    old=(R/'PLAN.json').read_bytes(); new=(D/'PLAN.revision7.json').read_bytes();p=json.loads(new)
    assert p['previous_plan_sha256']==hashlib.sha256(old).hexdigest()
    assert p['assignment_revision']==7 and len(p['pairs'])==len({t['id'] for t in p['pairs']})==6419
    a={t['id']:t for t in json.loads(old)['pairs']};b={t['id']:t for t in p['pairs']}
    assert a.keys()==b.keys()
    for i,t in a.items():assert {k:v for k,v in t.items() if k!='host'}=={k:v for k,v in b[i].items() if k!='host'}
    cache=[json.loads(s) for s in (H/'CACHE.jsonl').read_text().splitlines() if s.strip()]
    for c in cache:
        i=c['task_id']
        if i in a and a[i]['host']==h:assert b[i]['host']==h
    incoming={i for i,t in b.items() if t['host']==h and a[i]['host']!=h}
    rows=[json.loads(s) for s in (H/'journal.jsonl').read_text().splitlines() if s.strip()]
    assert not any(r['id'] in incoming and r['status'] not in ('temporarily_skipped_library','resource_wait') for r in rows)
    assert not incoming.intersection(c['task_id'] for c in cache)
    q=read(H/'QUARANTINE.json');assert not any(quarantine_key(b[i],q) for i in incoming)
    campaign=read(R/'CAMPAIGN.json')
    campaign.update(assignments=p['counts'],assignment_revision=7,assignment_updated_at=p['assignment_updated_at'],
        pending_at_revision7=read(D/'ASSIGNMENT_CHANGES.json')['ready_after'])
    for name, data in [('PLAN.json',new),('CAMPAIGN.json',(json.dumps(campaign,ensure_ascii=False,indent=2)+'\n').encode())]:
        backup=D/(h+'-before-'+name)
        with backup.open('xb') as f:f.write((R/name).read_bytes())
        tmp=R/(name+'.revision7-tmp');tmp.write_bytes(data);os.replace(tmp,R/name)
    result={'at':time.time(),'host':h,'assignment_revision':7,'plan_sha256':hashlib.sha256(new).hexdigest(),'incoming':len(incoming),'counts':p['counts']}
    (D/(h+'-PUBLISHED.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
