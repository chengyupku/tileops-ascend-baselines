"""Independent CPU check of publication values against immutable CI evidence."""
import collections
import argparse
import hashlib
import json
import math
from pathlib import Path

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
JOB = BASE/'ci-full-v3'


def read(path):
    return json.loads(path.read_text())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--export-dir',type=Path,default=BASE/'export-final-01')
parser.add_argument('--catalog-dir',type=Path,default=BASE/'current-baseline')
parser.add_argument('--saved-v2-only',action='store_true')
args = parser.parse_args()
snapshot_path = args.export_dir/'snapshot.json'
snapshot = read(snapshot_path)
catalog = read(args.catalog_dir/'BASELINE32_PRIVATE.json')
manifest = read(args.catalog_dir/'FROZEN_MANIFEST_SNAPSHOTS.json')
plan = read(JOB/'workspace/ELEMENTWISE_PLAN.json')
contracts = {(r['op'],r['case_id']):r for r in manifest['cases']}
cached = {tuple(r['key']):r for r in catalog['rows']}
assert len(cached) == 5550 and snapshot['manifest_cases'] == len(contracts) == 2208
rows = {(r['library'],r['op'],r['case']):r for r in snapshot['rows']}
assert len(rows) == len(snapshot['rows'])
native = {k:r for k,r in rows.items() if k[0] != 'torch-eager'}
assert set(native) == set(cached)
for key,row in native.items():
    assert math.isclose(row['latency_us'], cached[key]['admitted_latency_ms']*1000, rel_tol=1e-14)
    if args.saved_v2_only:
        original = cached[key]['timing']
        assert cached[key]['admission_metric'].startswith('sum_profiled_')
        assert original['measurement_protocol'] == 'ascend950-isolated-ffts-v2'
        assert math.isclose(row['latency_us'],sum(original['latency_samples_ms'])/15*1000,rel_tol=1e-12)
        assert row['latency_us'] == original['latency_ms']*1000
if args.saved_v2_only:
    gate=read(args.catalog_dir/'BASELINE_ADMISSION.json')
    assert gate['conservative_host_fallback']==0 and gate['device_sum_admitted']==5550
assert all(k[1] not in ('CausalConv1dFwdOp','CausalConv1dUpdateFwdOp') for k in native)
allowed = {'library','op','case','latency_us','samples','measurement_id','protocol','workload',
           'tileops_latency_us','ratio','tileops_samples','tileops_verified',
           'tileops_measurement_id','tileops_protocol'}
grouped = collections.defaultdict(list)
for key,row in rows.items():
    assert set(row) <= allowed, set(row)-allowed
    assert row['latency_us'] > 0 and math.isfinite(row['latency_us']) and row['samples'] == 15
    expected = json.loads(json.dumps(contracts[key[1:]], default=str), parse_constant=lambda value:value)
    assert row['workload'] == expected
    grouped[key[1:]].append(row)
timed = {}
for row in snapshot.get('tileops_rows', []):
    key = row['op'],row['case']
    assert key not in grouped and key not in timed
    assert row['verified'] and row['samples'] == 15
    timed[key] = row['latency_us']
eligible_count = 0
for entry in plan:
    directory = JOB/'measurements'/entry['op']
    pointer = directory/'LATEST_ATTEMPT.json'
    if pointer.exists():
        directory = JOB/'measurements'/read(pointer)['path']
    original = [json.loads(l) for l in (directory/'measurements.jsonl').read_text().splitlines()]
    supplement = read(directory/'EAGER_READMISSION.json') if (directory/'EAGER_READMISSION.json').exists() else {'cases':[]}
    overrides = {r['case_id']:r for r in supplement['cases']}
    for measured in original:
        key = entry['op'],measured['case_id']
        expected_us = sum(measured['latency_samples_ms'])/15*1000
        peers = grouped[key]
        for row in peers:
            assert row['tileops_verified'] is True and row['tileops_samples'] == 15
            assert math.isclose(row['tileops_latency_us'],expected_us,rel_tol=1e-12)
            assert math.isclose(row['ratio'],row['latency_us']/expected_us,rel_tol=1e-12)
        if peers:
            assert key not in timed
            timed[key] = peers[0]['tileops_latency_us']
        else:
            assert math.isclose(timed[key],expected_us,rel_tol=1e-12)
        eligible = overrides.get(measured['case_id'],{}).get('eager_correctness', measured['eager_correctness'])['eligible']
        eager_key = ('torch-eager',*key)
        assert (eager_key in rows) == eligible
        if eligible:
            eligible_count += 1
            assert math.isclose(rows[eager_key]['latency_us'],sum(measured['eager']['latency_samples_ms'])/15*1000,rel_tol=1e-12)
assert len(timed) == 527 and {op for op,_ in timed} == {p['op'] for p in plan}
result = dict(status='passed', native_pairs=len(native), fresh_eager_pairs=eligible_count,
              verified_operators=70, verified_cases=len(timed), manifest_cases=len(contracts),
              exact_native_values=True, exact_remote_tileops_values=True, ratios_verified=True,
              incompatible_historical_pairs_excluded=23, public_fields_allowlisted=True,
              all_historical_native_rows_use_saved_v2=args.saved_v2_only,
              snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest())
(args.export_dir/'INDEPENDENT_SNAPSHOT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
