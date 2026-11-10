"""Freeze public baseline measurements using CPU-only receipt checks."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root, host = Path(sys.argv[1]), sys.argv[2]
paths = [root / 'reports/baseline-cache-20261027' / host / 'CACHE.jsonl']
if host == 'server1':
    paths.append(root / 'reports/baseline-old1076-20261029/server1/CACHE.jsonl')
latest, rejected = {}, []
for path in paths:
    sidecar = path.parent / 'GUARD_BINDINGS.jsonl'
    bindings = {x['cache_key']: x for x in map(json.loads, sidecar.read_text().splitlines())} if sidecar.exists() else {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            assert r['status'] == 'cached' and r['performance_valid'] is True
            cpath = Path(r['correctness_receipt'])
            raw = cpath.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == r['correctness_sha256']
            c = json.loads(raw)
            v = c['verification']
            assert c['status'] == 'verified' and v['checked_outputs'] > 0
            assert v['checked_outputs'] == v['total_outputs'] and not v.get('unchecked_reason')
            assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
            assert [c['library'], c['op'], c['case_id']] == r['key']
            graw = Path(r['guard_receipt']).read_bytes()
            g = json.loads(graw)
            assert g['phase'] == 'finished' and g['performance_valid'] is True
            assert g['workload_exit_code'] == 0 and not g['conflict_detected'] and not g['visibility_lost']
            assert g['owner'] == f"baseline-cache-20261027/{host}/{r['task_id']}"
            binding = r.get('guard_binding') or bindings.get(r['cache_key'])
            assert binding, 'missing guard binding'
            assert hashlib.sha256(graw).hexdigest() == binding['receipt_sha256']
            p = r['identity']['protocol']
            assert (p['warmup'], p['repeats'], p['cache_flush_mib']) == (3, 15, 1024)
            timing = r.get('timing') or {}
            device_samples = timing.get('latency_samples_ms') or []
            has_device = (len(device_samples) == 15 and
                          all(math.isfinite(x) and x > 0 for x in device_samples) and
                          math.isfinite(timing.get('latency_ms', float('nan'))) and
                          math.isclose(statistics.mean(device_samples), timing['latency_ms'], rel_tol=1e-10, abs_tol=1e-10))
            for samples, mean in [(r['end_to_end_samples_ms'], r['end_to_end_mean_ms'])]:
                assert len(samples) == 15 and all(math.isfinite(x) and x > 0 for x in samples)
                assert math.isclose(statistics.mean(samples), mean, rel_tol=1e-10, abs_tol=1e-10)
            metric = 'device_latency_ms' if has_device else 'end_to_end_mean_ms'
            ms = timing['latency_ms'] if has_device else r['end_to_end_mean_ms']
            assert metric in ('device_latency_ms', 'end_to_end_mean_ms') and math.isfinite(ms) and ms > 0
            item = dict(host=host, library=r['key'][0], op=r['key'][1], case=r['key'][2],
                        latency_us=ms * 1000, metric=metric, original_metric=r['comparison_metric'], samples=15,
                        measured_at=r['finished_at'], cache_key=r['cache_key'],
                        cohort=r['cohort_key'], protocol=p['id'],
                        workload=r['identity']['case'], tileops_latency_us=None, ratio=None)
            key = tuple(r['key'])
            if key not in latest or item['measured_at'] > latest[key]['measured_at']:
                latest[key] = item
        except Exception as exc:
            rejected.append({'id': r.get('task_id') if 'r' in locals() else None,
                             'reason': type(exc).__name__ + ': ' + str(exc)})
print(json.dumps(dict(host=host, rows=list(latest.values()), rejected=rejected), separators=(',', ':')))
