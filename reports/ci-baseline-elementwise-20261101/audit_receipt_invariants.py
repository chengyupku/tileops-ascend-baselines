"""Additional CPU-only execution and timing consistency checks."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

root, host, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
checks = []
for campaign in ('baseline-cache-20261027', 'baseline-old1076-20261029', 'baseline-retest32-20261030'):
    for raw in (root / 'reports' / campaign / host / 'CACHE.jsonl').open():
        row = json.loads(raw)
        errors = []
        t = row['timing']
        samples = t['latency_samples_ms']
        def check(ok, reason):
            if not ok:
                errors.append(reason)
        check(math.isclose(t['profile_kernel_us'], t['latency_ms'] * 1000, rel_tol=1e-9), 'unit conversion mismatch')
        check(math.isfinite(t['profile_kernel_count']) and t['profile_kernel_count'] > 0, 'no profiled kernels')
        check(t.get('profile_excluded_kernel_count', 0) == 0, 'flush leaked into timed window')
        for field, expected in [('latency_min_ms', min(samples)), ('latency_max_ms', max(samples)),
                                ('latency_median_ms', statistics.median(samples)),
                                ('latency_stdev_ms', statistics.stdev(samples))]:
            check(math.isclose(t[field], expected, rel_tol=1e-9, abs_tol=1e-12), field + ' mismatch')
        receipt = json.loads(Path(row['correctness_receipt']).read_text())
        calls = receipt.get('observed_calls', {})
        for actor in ('candidate', 'reference'):
            observations = calls.get(actor, [])
            check(bool(observations) and all(x.get('returned') for x in observations), actor + ' not observed returning')
        binding = row.get('guard_binding')
        if binding:
            check(binding['before']['files'] == binding['after']['files'], 'guard component changed during measurement')
            check(binding['before']['stamps'] == binding['after']['stamps'], 'guard stamp changed during measurement')
        checks.append(dict(key=row['key'], cache_key=row['cache_key'], errors=errors))
report = dict(checked=len(checks), passed=sum(not r['errors'] for r in checks),
              errors=dict(collections.Counter(e for r in checks for e in r['errors'])),
              device_runs=0, rows=checks)
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k != 'rows'}))
