"""Create a host-free website snapshot from fully verified physical32 results.

Run on the CI host with its standard-library Python. This never imports torch or
accesses the device; all retained raw profiler windows are replayed on CPU.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def workload(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value > 0 else '-Infinity'
    if isinstance(value, dict):
        return {k: workload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [workload(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-op', action='append', default=[])
    p.add_argument('--require-all', action='store_true')
    p.add_argument('--catalog-dir', type=Path)
    p.add_argument('--public-regressions', type=Path)
    a = p.parse_args()
    job = a.job.resolve()
    root = job / 'workspace'
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / 'tools'))
    spec = importlib.util.spec_from_file_location('ci_controller', root / 'tools/ci_elementwise32.py')
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    frozen = controller.fingerprint()
    gate = read(root / 'BASELINE_ADMISSION.json')
    assert gate['status'] == 'passed' and sha(root / 'BASELINE32_PRIVATE.json') == gate['catalog_sha256']
    measured_catalog = read(root / 'BASELINE32_PRIVATE.json')
    publication = (a.catalog_dir or job.parent / 'current-baseline').resolve()
    current_gate = read(publication / 'BASELINE_ADMISSION.json')
    assert current_gate['status'] == 'passed'
    assert current_gate.get('metric_policy_id') == 'saved-v2-device-sum-user-20261101', 'Current user policy requires saved v2 device sums; H fallback is not allowed'
    assert current_gate['conservative_host_fallback'] == 0
    assert sha(publication / 'BASELINE32_PRIVATE.json') == current_gate['catalog_sha256']
    assert sha(publication / 'FROZEN_MANIFEST_SNAPSHOTS.json') == current_gate['manifest_snapshots_sha256']
    manifest = read(publication / 'FROZEN_MANIFEST_SNAPSHOTS.json')
    for path, digest in manifest['sources'].items():
        assert sha(root / 'TileOPs' / path) == digest, f'Manifest source changed: {path}'
    manifest_cases = {(c['op'], c['case_id']): c for c in manifest['cases']}
    assert len(manifest_cases) == current_gate['manifest_cases']
    catalog = read(publication / 'BASELINE32_PRIVATE.json')
    assert len(catalog['rows']) == current_gate['total_unique_pairs']
    assert all(row['case'] == manifest_cases[tuple(row['key'][1:])] for row in catalog['rows'])
    plan = read(root / 'ELEMENTWISE_PLAN.json')
    cases = {(op['op'], cid): case for op in plan for cid, case in op['case_snapshots'].items()}
    assert all(manifest_cases[key] == case for key, case in cases.items())
    measured_target = {tuple(r['key']):r for r in measured_catalog['rows'] if tuple(r['key'][1:]) in cases}
    publication_target = {tuple(r['key']):r for r in catalog['rows'] if tuple(r['key'][1:]) in cases}
    if current_gate.get('metric_policy_id') == 'saved-v2-device-sum-user-20261101':
        # This policy transition changes selection of an already retained metric,
        # never the frozen samples, numerical evidence, workload or measured code.
        assert current_gate['historical_parent_catalog_sha256'] == gate['catalog_sha256']
        assert current_gate['conservative_host_fallback'] == 0
        assert current_gate['device_sum_admitted'] == len(catalog['rows'])
        assert publication_target.keys() == measured_target.keys()
        ignored = {'admitted_latency_ms', 'admission_metric', 'device_sum_quarantined'}
        for key, row in publication_target.items():
            original = measured_target[key]
            assert {k:v for k,v in row.items() if k not in ignored} == {
                k:v for k,v in original.items() if k not in ignored}
        for row in catalog['rows']:
            assert row['admitted_latency_ms'] == row['timing']['latency_ms']
            assert math.isclose(statistics.mean(row['timing']['latency_samples_ms']), row['admitted_latency_ms'], rel_tol=1e-9)
            assert row['admission_metric'] in {
                'sum_profiled_single_core_role_kernel_durations_mean15',
                'sum_profiled_v2_kernel_record_durations_mean15'}
    else:
        assert publication_target == measured_target, 'Publication changed the benchmark comparison baselines'
    public_rows = []
    indexed = collections.defaultdict(list)
    for row in catalog['rows']:
        library, op, case = row['key']
        if (op, case) in cases:
            assert row['case'] == cases[(op, case)], 'Cached case contract differs'
        public = dict(library=library, op=op, case=case,
                      latency_us=row['admitted_latency_ms']*1000, samples=15,
                      measurement_id=identifier([row['cache_key'], row['admission_metric']]),
                      protocol=row['protocol']['id'], workload=workload(row['case']),
                      tileops_latency_us=None, ratio=None)
        public_rows.append(public)
        indexed[(op, case)].append(public)
    qualified = []
    missing = []
    bad_eager = []
    performance = []
    tileops_only = []
    excluded = set(a.exclude_op)
    for entry in plan:
        op = entry['op']
        directory = job / 'measurements' / op
        pointer = directory / 'LATEST_ATTEMPT.json'
        if pointer.exists():
            directory = job / 'measurements' / read(pointer)['path']
        receipt_path = directory / 'receipt.json'
        if op in excluded or not receipt_path.exists():
            missing.append(dict(op=op, reason='excluded pending independent regression' if op in excluded else 'no complete receipt'))
            continue
        receipt = read(receipt_path)
        if receipt.get('status') != 'passed':
            missing.append(dict(op=op, reason='failed', errors=receipt.get('errors')))
            continue
        if receipt['sources'] != frozen:
            proof = read(directory / 'READMISSION.json')
            assert proof['status'] == 'passed' and proof['current_sources'] == frozen
            assert proof['original_receipt_sha256'] == sha(receipt_path)
            assert proof['production_kernel_changes'] == []
        guard = read(directory / 'guard.json')
        assert guard['phase'] == 'finished' and guard['performance_valid'] is True
        assert guard['workload_exit_code'] == 0 and not guard.get('conflict_detected') and not guard.get('visibility_lost')
        rows = [json.loads(line) for line in (directory / 'measurements.jsonl').read_text().splitlines()]
        controller.validate_rows(rows, entry['case_ids'])
        supplemental_eager = {}
        supplement_path = directory / 'EAGER_READMISSION.json'
        if supplement_path.exists():
            supplement = read(supplement_path)
            assert op == 'DropoutFwdOp' and supplement['status'] == 'passed'
            assert supplement['sources'] == frozen
            assert supplement['original_measurements_sha256'] == sha(directory / 'measurements.jsonl')
            assert supplement['original_receipt_sha256'] == sha(receipt_path)
            assert supplement['production_kernel_changes'] == supplement['timing_changes'] == []
            assert sha(supplement['guard_path']) == supplement['guard_sha256']
            assert sha(supplement['validation_path']) == supplement['validation_sha256']
            fresh_guard = read(supplement['guard_path'])
            assert fresh_guard['owner'] == supplement['command'][supplement['command'].index('--owner')+1]
            assert fresh_guard['phase'] == 'finished' and fresh_guard['performance_valid'] is True
            assert fresh_guard['workload_exit_code'] == 0 and not fresh_guard.get('conflict_detected') and not fresh_guard.get('visibility_lost')
            validation = read(supplement['validation_path'])
            assert validation['status'] == 'passed' and validation['correctness_only'] is True
            assert validation['original_timings_not_modified'] is True
            assert validation['cases'] == supplement['cases']
            supplemental_eager = {c['case_id']:c for c in supplement['cases']}
            assert set(supplemental_eager) == set(entry['case_ids'])
        for row in rows:
            key = (op, row['case_id'])
            device_us = row['latency_ms']*1000
            measurement_id = identifier([op, row['case_id'], row['source_hash'],
                                         row['latency_samples_ms'],
                                         [trace['profile_trace_sha256'] for trace in row['retained_profiles']]])
            eager_check = row['eager_correctness']
            if row['case_id'] in supplemental_eager:
                fresh = supplemental_eager[row['case_id']]
                assert not eager_check['eligible'] and 'same device' in eager_check['error']
                assert fresh['input_digest'] == row['input_digest'] and fresh['input_unchanged']
                assert fresh['negative_controls_rejected'] == ['all_zero', 'wrong_scale', 'extra_output']
                eager_check = fresh['eager_correctness']
                assert eager_check['eligible'] is True
            if eager_check['eligible']:
                verification = eager_check['verification']
                assert verification['checked_outputs'] == verification['total_outputs'] > 0
                eager = row['eager']
                candidate = dict(library='torch-eager', op=op, case=row['case_id'],
                                 latency_us=eager['latency_ms']*1000, samples=15,
                                 measurement_id=identifier([measurement_id, 'eager', eager['latency_samples_ms']]),
                                 protocol=eager['measurement_protocol'], workload=workload(cases[key]),
                                 tileops_latency_us=None, ratio=None)
                public_rows.append(candidate)
                indexed[key].append(candidate)
            else:
                bad_eager.append(dict(op=op, case=row['case_id'], evidence=eager_check))
            for candidate in indexed[key]:
                candidate.update(tileops_latency_us=device_us,
                                 ratio=candidate['latency_us']/device_us,
                                 tileops_samples=15, tileops_verified=True,
                                 tileops_measurement_id=measurement_id,
                                 tileops_protocol=row['measurement_protocol'])
            if not indexed[key]:
                tileops_only.append(dict(op=op, case=row['case_id'], latency_us=device_us,
                                         samples=15, verified=True, workload=workload(cases[key]),
                                         measurement_id=measurement_id,
                                         protocol=row['measurement_protocol']))
            best = min((r['latency_us'] for r in indexed[key]), default=None)
            performance.append(dict(op=op, case=row['case_id'], tileops_us=device_us,
                                    fastest_admitted_baseline_us=best,
                                    ratio=best/device_us if best is not None else None,
                                    meets_80_percent=best/device_us >= .8 if best is not None else None))
        qualified.append(op)
    if a.require_all:
        assert not missing and len(qualified) == 70 and len(performance) == 527, missing
        public = a.public_regressions or job.parent / 'public-regressions-corrected-full-01'
        regression = read(public / 'receipt.json')
        assert regression['status'] == 'passed' and regression['sources'] == frozen
        regression_guard = read(public / 'guard.json')
        assert regression_guard['phase'] == 'finished' and regression_guard['performance_valid'] is True
        assert regression_guard['workload_exit_code'] == 0
        assert not regression_guard.get('conflict_detected') and not regression_guard.get('visibility_lost')
        tests = list(ET.parse(public / 'junit.xml').iter('testcase'))
        assert len(tests) == 776 and all(
            c.find('failure') is None and c.find('error') is None and c.find('skipped') is None
            for c in tests), 'Complete corrected public regression has not passed'
    snapshot = dict(schema=2, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    manifest_cases=len(manifest_cases), rows=public_rows, tileops_rows=tileops_only)
    raw = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
    (output / 'snapshot.json').write_text(raw)
    report = dict(qualified_operators=len(qualified), qualified_cases=len(performance),
                  baseline_rows=len(public_rows), original_cached_baselines=len(catalog['rows']),
                  tileops_only_cases=len(tileops_only), missing=missing, ineligible_eager=bad_eager,
                  below_80_percent=[p for p in performance if p['meets_80_percent'] is False],
                  performance=performance, private_source_revision=subprocess.check_output(
                      ['git', '-C', str(root/'TileOPs'), 'rev-parse', 'HEAD'], text=True).strip(),
                  snapshot_sha256=sha(output/'snapshot.json'),
                  publication_catalog_sha256=current_gate['catalog_sha256'],
                  excluded_historical_contract_pairs=current_gate['historical_contract_pairs'],
                  frozen_manifest_sha256=current_gate['manifest_snapshots_sha256'],
                  metric_policy_id=current_gate.get('metric_policy_id', 'historical-H-fallback'),
                  qualification=('Case correctness and exclusive valid profiling verified on physical32 CI. '
                      'Historical native baselines use original saved v2 device sums under explicit revised user policy; '
                      'H is excluded, known v2 bias retained pending future native baseline v3 remeasurement. '
                      'Already measured v3 TileOPs/eager results are preserved.'
                      if current_gate.get('metric_policy_id') == 'saved-v2-device-sum-user-20261101' else
                      'Historical export policy: saved H fallback; superseded by revised v2-only policy.'))
    (output / 'PRIVATE_REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = {k:v for k,v in report.items() if k not in
               ('performance', 'below_80_percent', 'ineligible_eager', 'missing')}
    summary.update(missing_operators=len(missing), below_80_percent_cases=len(report['below_80_percent']),
                   ineligible_eager_cases=len(bad_eager))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
