"""Admit the union of independently audited remote32 cache records."""
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data, *, strict=True):
    raw = (json.dumps(data, ensure_ascii=False, allow_nan=not strict, separators=(',', ':')) + '\n').encode()
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def public_workload(value):
    # Infinity is a valid norm parameter. Keep the original private Python JSON
    # evidence, and use a string for standards-compliant downloadable JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {k: public_workload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_workload(v) for v in value]
    return value


def main():
    check = read(ROOT / 'server1-last-check.json')['result']
    assert check['complete'] and check['cached'] == check['assigned'] == 1187
    assert not check['missing'] and not check['extra']
    inputs = [ROOT / 'server1-catalog-resolved.json', ROOT / 'server2-catalog-resolved.json']
    rows = [row for path in inputs for row in read(path)]
    assert not any(row['errors'] for row in rows), 'Unresolved cache audit failure'
    assert len(rows) == len({tuple(row['key']) for row in rows}) == 5573
    expected = {tuple(p['key']) for p in read(WORKSPACE / 'reports/baseline-retest32-20261030/PLAN.json')['pairs']}
    actual = {tuple(row['key']) for row in rows if row['campaign'] == 'baseline-retest32-20261030'}
    assert len(expected) == 2434 and actual == expected
    assert all(row['environment']['device_name'] == 'Ascend950DT_9582' for row in rows)
    mixed_fallback = 0
    for row in rows:
        mixed = row['timing'].get('profile_aic_cycles', 0) > 0 and row['timing'].get('profile_aiv_cycles', 0) > 0
        if mixed:
            mixed_fallback += 1
            row['admitted_latency_ms'] = row['end_to_end_mean_ms']
            row['admission_metric'] = 'host_roundtrip_with_device_synchronize_mean15'
            row['device_sum_quarantined'] = 'Potential duplicate AIC/AIV roles of one kernel; original per-task trace unavailable. Conservative H fallback under prior user authorization.'
        else:
            row['admitted_latency_ms'] = row['timing']['latency_ms']
            row['admission_metric'] = 'sum_profiled_single_core_role_kernel_durations_mean15'
    frozen = datetime.datetime.now(datetime.timezone.utc).isoformat()
    catalog = dict(schema=1, frozen_at=frozen, hardware=dict(model='Ascend950DT_9582', cube=32, vector=64),
                   protocol='ascend950-isolated-ffts-v2', rows=rows)
    digest = dump(ROOT / 'BASELINE32_PRIVATE.json', catalog, strict=False)
    gate = dict(schema=1, status='passed', server1_complete=True, server1_retested=1187,
                server2_retested=1247, local36_replaced=2434, total_unique_pairs=len(rows),
                device_sum_admitted=len(rows)-mixed_fallback, conservative_host_fallback=mixed_fallback,
                metric_policy='Prefer valid saved device kernel sums; where usable per-kernel D cannot be recovered, use complete saved synchronized H samples as user authorized; keep provenance private.',
                catalog_sha256=digest, audit_inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                qualification='Original full correctness, exact case, source/native closure, physical32 hardware, exclusive guard and15 independent profiler samples verified. Historical exact-source resolution is retained separately.')
    dump(ROOT / 'BASELINE_ADMISSION.json', gate)
    public = []
    for row in rows:
        library, op, case = row['key']
        public.append(dict(library=library, op=op, case=case,
                           latency_us=row['admitted_latency_ms'] * 1000, samples=15,
                           measurement_id=row['cache_key'], protocol=row['protocol']['id'],
                           workload=public_workload(row['case']), tileops_latency_us=None, ratio=None))
    old = read(WORKSPACE / '.worktrees/TileOPs-site-20261101/data/baseline-cache/snapshot.json')
    dump(ROOT / 'BASELINE32_PUBLIC.json', dict(schema=2, frozen_at=frozen, manifest_cases=old['manifest_cases'], rows=public))
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
