"""Prepare sanitized publication; keep timing selection evidence private."""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

root = Path('/root/dyq/workspace')
report = root / 'reports/baseline-site-20261029'
manifest = json.loads((root / '.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines/cases.json').read_text())
contracts = {(r['op'], r['case_id']): r for r in manifest['cases']}
latest = {}
for host in ['local', 'server1', 'server2']:
    data = json.loads((report / f'preferred-{host}.json').read_text())
    assert not data['rejected'], data['rejected']
    for row in data['rows']:
        assert row['workload'] == contracts[row['op'], row['case']]
        key = (row['library'], row['op'], row['case'])
        # Prefer any available device measurement, then newest, never fastest.
        priority = (row['metric'] == 'device_latency_ms', row['measured_at'])
        if key not in latest or priority > (latest[key]['metric'] == 'device_latency_ms', latest[key]['measured_at']):
            latest[key] = row
rows = [latest[k] for k in sorted(latest)]
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
audit = dict(frozen_at=now, policy='Prefer saved per-call profiler kernel-duration sums, including overlapping kernels. Otherwise use saved host wall time. No remeasurement.',
             selected_metrics=dict(collections.Counter(r['metric'] for r in rows)),
             original_metrics=dict(collections.Counter(r['original_metric'] for r in rows)),
             measurements=len(rows), operators=len({r['op'] for r in rows}),
             workloads=len({(r['op'], r['case']) for r in rows}), rows=rows)
(report / 'preferred-selection-audit.json').write_text(json.dumps(audit, indent=2))

def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value

public = [clean({k:v for k,v in r.items() if k not in {'host','cohort','metric','original_metric'}}) for r in rows]
snapshot = dict(schema='ascend950-baseline-progress-v3', frozen_at=now, manifest_cases=len(contracts), rows=public)
path = root / '.worktrees/TileOPs-site/data/baseline-cache/snapshot.json'
path.write_text(json.dumps(snapshot, separators=(',', ':'), allow_nan=False) + '\n')
print(json.dumps({k:v for k,v in audit.items() if k != 'rows'}))
print('sha256', hashlib.sha256(path.read_bytes()).hexdigest())
