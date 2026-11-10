"""Publish a prepared ownership revision only while this host is stopped."""
import collections
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

host, stage = sys.argv[1:]
assert host in ('local', 'server2') and stage in ('pilot', 'half')
r = Path('reports/baseline-cache-20261027')
d = r / 'balance-half-20261028'
assert (r / host / 'STOP').exists()
with (r / host / 'supervisor.lock').open('a') as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # A STOP barrier permits a handoff of other, unstarted keys while the
        # current unrelated pair keeps its existing resource queue position.
        assert host == 'local' and stage == 'half'
        assert (r/host/'STOP').read_text() == 'balance-half-20261028/final-ownership-handoff\n'
        live=json.loads((r/host/'LIVE.json').read_text())
        assert live['phase']=='running' and live['key'][0]!='sgl-kernel-npu'
    old = (r/'PLAN.json').read_bytes()
    new = (d/('PLAN.'+stage+'.json')).read_bytes()
    plan = json.loads(new)
    assert plan['previous_plan_sha256'] == hashlib.sha256(old).hexdigest()
    assert len(plan['pairs']) == len({t['id'] for t in plan['pairs']}) == 6419
    old_tasks = {t['id']: t for t in json.loads(old)['pairs']}
    new_tasks = {t['id']: t for t in plan['pairs']}
    for task_id, task in old_tasks.items():
        assert {k:v for k,v in task.items() if k!='host'} == {k:v for k,v in new_tasks[task_id].items() if k!='host'}
    cached = [json.loads(s) for s in (r/host/'CACHE.jsonl').read_text().splitlines()]
    assert all(new_tasks[c['task_id']]['host'] == host for c in cached if c['task_id'] in old_tasks and old_tasks[c['task_id']]['host'] == host)
    for name in ('PLAN.json', 'CAMPAIGN.json'):
        backup=d/(host+'-'+stage+'-before-'+name)
        assert not backup.exists()
        backup.write_bytes((r/name).read_bytes())
    campaign=json.loads((r/'CAMPAIGN.json').read_text())
    campaign.update(assignments=plan['counts'], assignment_revision=plan['assignment_revision'],
                    assignment_updated_at=plan['assignment_updated_at'])
    if stage == 'half':
        proof=json.loads((d/'half-changes.json').read_text())
        campaign['equal_pending_at_publish']=proof['remaining_after']
    for name, data in [('PLAN.json',new), ('CAMPAIGN.json',(json.dumps(campaign,ensure_ascii=False,indent=2)+'\n').encode())]:
        tmp=r/(name+'.publish-tmp')
        tmp.write_bytes(data)
        os.replace(tmp,r/name)
    print(json.dumps({'host':host,'stage':stage,'revision':plan['assignment_revision'],
                      'plan_sha256':hashlib.sha256(new).hexdigest(),'counts':plan['counts']}))
