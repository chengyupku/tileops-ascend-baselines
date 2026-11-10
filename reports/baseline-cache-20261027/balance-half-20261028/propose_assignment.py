"""CPU-only proposal, from stopped-host snapshots. Does not publish ownership."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

root = Path('reports/baseline-cache-20261027')
report = root / 'balance-half-20261028'
stage, snapshot_label = sys.argv[1:]
assert stage in ('pilot', 'half')
source = (root / 'PLAN.json').read_bytes()
plan = json.loads(source)
snapshots = {host: json.loads((report / (host + '-' + snapshot_label + '.json')).read_text())
             for host in ('local', 'server2')}
for host, snapshot in snapshots.items():
    assert snapshot['host'] == host
    if snapshot['identities_match'].get('identity', False):
        assert stage == 'half' and host == 'local'
        assert snapshot['control_stop'] == 'balance-half-20261028/final-ownership-handoff\n'
        assert snapshot['live']['phase'] == 'running'
        assert snapshot['live']['key'][0] != 'sgl-kernel-npu'
    else:
        assert snapshot['live']['phase'] in ('stopped_by_control', 'pilot_complete', 'sweep_complete')
ready = {h: {t['id']: t for t in s['ready']} for h, s in snapshots.items()}
cached = set().union(*(set(s['cache_ids']) for s in snapshots.values()))
if stage == 'pilot':
    selected = [p['id'] for p in json.loads((report / 'PILOTS.proposed.json').read_text())]
else:
    count = (len(ready['local']) - len(ready['server2'])) // 2
    assert count >= 0
    keep_local = {t['id'] for t in json.loads((report/'KEEP_LOCAL_DEPLOYMENT.json').read_text())}
    selected = [i for i, t in ready['local'].items() if t['key'][0] == 'sgl-kernel-npu' and i not in keep_local][:count]
    assert len(selected) == count, 'Insufficient validated deployment cohort; do not assign unavailable tasks'
assert len(selected) == len(set(selected))
assert set(selected) <= ready['local'].keys()
assert not (set(selected) & cached)
assert all(t['key'] != snapshots['local']['live'].get('key') for t in ready['local'].values() if t['id'] in selected)
changes = []
for t in plan['pairs']:
    if t['id'] in selected:
        assert t['host'] == 'local'
        changes.append({'id': t['id'], 'key': t['key'], 'from': 'local', 'to': 'server2'})
        t['host'] = 'server2'
assert len(changes) == len(selected)
assert len(plan['pairs']) == len({t['id'] for t in plan['pairs']}) == 6419
# Put the four real validation pairs first, only for the bounded pilot sweep.
if stage == 'pilot':
    order = {task_id: pos for pos, task_id in enumerate(selected)}
    plan['pairs'].sort(key=lambda t: order.get(t['id'], len(selected)))
at = datetime.datetime.now(datetime.timezone.utc).isoformat()
plan.update(assignment_revision=plan['assignment_revision']+1,
            counts=dict(collections.Counter(t['host'] for t in plan['pairs'])),
            assignment_revision_at=at, assignment_updated_at=at,
            previous_plan_sha256=hashlib.sha256(source).hexdigest(),
            assignment_evidence='balance-half-20261028/'+stage+'-changes.json',
            assignment_reason='User requested equal actual remaining runnable work; original ARM64 SGL deployment')
before = {h: len(v) for h, v in ready.items()}
after = {'local': before['local']-len(selected), 'server2': before['server2']+len(selected)}
if stage == 'half':
    assert abs(after['local']-after['server2']) <= 1
evidence = {'at': at, 'stage': stage, 'snapshot_label': snapshot_label,
            'remaining_before': before, 'remaining_after': after,
            'moved': len(selected), 'changes': changes, 'cached_ids_unchanged': len(cached)}
for name, value in [(stage+'-changes.json', evidence), ('PLAN.'+stage+'.json', plan)]:
    p=report/name
    assert not p.exists(), 'Keep previous proposal evidence'
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in evidence.items() if k!='changes'}))
