import copy

from baseline_release import merge_latest_release
from nightly_snapshot import build_cache


def row(n, library='torch-eager', latency=10):
    return dict(library=library, op='ExampleOp', case='same',
                workload={'params': {'n': n}, 'family': 'elementwise'},
                latency_us=latency, measurement_id=str(n), protocol='v3',
                samples=15, tileops_latency_us=None, ratio=None)


def test_old_contract_nightly_never_attaches_ratio_to_new_contract():
    latest, old = row(64), row(32)
    base = dict(baseline_release='v2', rows=[latest], historical_rows=[old], frozen_at='new')
    measured = dict(op='ExampleOp', case_id='same', params={'n': 32},
                    workload=old['workload'], measurement={'latency_ms': .005})
    result = build_cache(base, [measured], 'a' * 40, 'later')
    assert result['rows'][0]['tileops_latency_us'] is None
    assert result['historical_rows'][0]['ratio'] == 2
    assert result['tileops_rows'] == []
    assert base['historical_rows'][0]['tileops_latency_us'] is None


def test_unmatched_contract_keeps_evidence_separate():
    base = dict(baseline_release='v2', rows=[row(64)], frozen_at='new')
    measured = dict(op='ExampleOp', case_id='same', params={'n': 32},
                    workload=row(32)['workload'], measurement={'latency_ms': .005})
    result = build_cache(base, [measured], 'a' * 40, 'later')
    assert result['rows'][0]['ratio'] is None
    assert result['tileops_rows'] == []
    assert result['historical_tileops_rows'][0]['latency_us'] == 5


def test_older_nightly_cannot_roll_back_torch_or_native_release():
    rows = [row(64), row(64, 'native', 8)]
    published = dict(rows=rows, historical_rows=[row(32)], frozen_at='2026-11-07')
    official = dict(rows=[row(32)], frozen_at='2026-11-06')
    official['rows'][0].update(tileops_latency_us=5, tileops_date='2026-11-06')
    release = dict(count=2, measurements=[{k: r[k] for k in
                   ('library', 'op', 'case', 'measurement_id', 'protocol')} for r in rows])
    result = merge_latest_release(official, published, release)
    assert result['rows'] == rows
    assert result['historical_rows'][0]['ratio'] == 2
    matching = copy.deepcopy(official)
    matching['rows'][0]['workload'] = rows[0]['workload']
    result = merge_latest_release(matching, published, release)
    assert [r['ratio'] for r in result['rows']] == [2, 1.6]
    assert result['frozen_at'] == '2026-11-07'
