import collections
import json
import pathlib
import sys
import time

host = sys.argv[1]
root = pathlib.Path('reports/baseline-cache-20261027')
out = root / 'inspection-20261028T201544'
read = lambda p: json.loads(p.read_text())
holds = read(out / (host + '-FP16_HOLDS.json'))
ids = {r['id'] for r in holds['added']}
rows = [json.loads(s) for s in (root / host / 'journal.jsonl').read_text().splitlines() if s.strip()]
recent = [r for r in rows if r.get('at', 0) >= holds['at']]
held_recent = [r for r in recent if r['id'] in ids]
live = read(root / host / 'LIVE.json')
result = {
    'at': time.time(), 'host': host, 'new_holds': len(ids),
    'holds_present_in_live_projection': sum('pair:' + i in live.get('quarantine', {}) for i in ids),
    'held_consumption': dict(collections.Counter(r['status'] for r in held_recent)),
    'held_device_attempts_after_action': [r for r in held_recent if r.get('wall_seconds') is not None],
    'production_after_action': dict(collections.Counter(r['status'] for r in recent)),
    'current_live': {k: live.get(k) for k in ['at', 'identity', 'phase', 'key', 'pair_phase', 'guard_identity']},
}
if host == 'local':
    old = read(root / 'inspection-20261028T161544/ADMISSION_HOLDS.json')
    old_ids = {r['id'] if isinstance(r, dict) else r for r in old['added']}
    after_old = [r for r in rows if r.get('at', 0) >= old['at'] and r['id'] in old_ids]
    result['previous_356_hold_consumption'] = dict(collections.Counter(r['status'] for r in after_old))
    result['previous_356_device_attempts'] = [r for r in after_old if r.get('wall_seconds') is not None]
(out / (host + '-EFFECT.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False))
