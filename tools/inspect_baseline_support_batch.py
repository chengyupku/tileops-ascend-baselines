"""CPU-only aggregate and concrete failed-command evidence for the support sweep."""
import collections
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
host = sys.argv[1]
b = root / 'reports/baseline-cache-20261027'
d = b / host
live = json.loads((d / 'LIVE.json').read_text())
rows = [json.loads(line) for line in (d / 'journal.jsonl').read_text().splitlines()]
scope = {x['id'] for x in json.loads((b / 'support-run-20261029/SCOPE.json').read_text())}
owned = {x['id'] for x in json.loads((b/'PLAN.json').read_text())['pairs'] if x['host'] == host and x['id'] in scope}
# Select ownership first, then latest journal row across all repair revisions.
# New environment/performance revisions need not begin with support-.
rows = [r for r in rows if r['id'] in owned]
latest = {r['id']: r for r in rows}
failures = []
supported = []
for row in latest.values():
    out = Path(row.get('measurement', '')).parent
    checkpath = out / 'correctness.json'
    check = json.loads(checkpath.read_text()) if checkpath.is_file() else {}
    if check.get('status') == 'verified':
        supported.append({'id': row['id'], 'key': row['key'], 'bucket': 'baseline支持',
                          'performance_cached': row['status'] == 'cached',
                          'correctness_receipt': str(checkpath), 'journal_status': row['status']})
    if row['status'] != 'cached':
        log = out / 'run.log'
        failures.append({**row, 'checker_status': check.get('status'),
                         'checker_error': check.get('error', check.get('reason')),
                         'checker_stage': check.get('stage'), 'api': check.get('api'),
                         'verification': check.get('verification'),
                         'log_tail': log.read_text(errors='replace')[-6000:] if log.exists() else None})
identity = live.get('identity', {})
p = Path('/proc') / str(identity.get('pid', 0))
alive = False
if p.exists():
    alive = (int((p / 'stat').read_text().rsplit(')', 1)[1].split()[19]) == identity.get('start_ticks')
             and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == identity.get('boot_id'))
result = {'at': time.time(), 'host': host, 'identity_alive': alive,
          'live': {k: v for k, v in live.items() if k != 'quarantine'},
          'new_counts': dict(collections.Counter(r['status'] for r in latest.values())),
          'supported': supported, 'failures': failures}
out = b / 'support-run-20261029' / (host + '-TRIAGE.json')
tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n'); tmp.replace(out)
print(json.dumps({'host': host, 'alive': alive, 'counts': result['new_counts'],
                  'phase': live.get('pair_phase'), 'key': live.get('key'),
                  'failures': [{'id': r['id'], 'key': r['key'], 'checker_status': r['checker_status'],
                                'error': str(r['checker_error'] or r.get('error'))[:500]} for r in failures]}, ensure_ascii=False))
