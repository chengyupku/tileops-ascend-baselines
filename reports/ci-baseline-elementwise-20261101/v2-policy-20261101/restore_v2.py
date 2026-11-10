"""Select saved v2 device sums under the user's revised publication policy."""
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

BASE = Path(__file__).resolve().parent
OLD = BASE.parent/'current-contract-baseline'
OUT = BASE/'catalog'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data,separators=(',', ':'))+'\n')
    return sha(path)


gate = json.loads((OLD/'BASELINE_ADMISSION.json').read_text())
assert sha(OLD/'BASELINE32_PRIVATE.json') == gate['catalog_sha256']
catalog = json.loads((OLD/'BASELINE32_PRIVATE.json').read_text())
OUT.mkdir(exist_ok=False)
rows = copy.deepcopy(catalog['rows'])
changed = []
for old,row in zip(catalog['rows'],rows,strict=True):
    timing = row['timing']
    values = timing['latency_samples_ms']
    assert row['protocol']['id'] == 'ascend950-isolated-ffts-v2'
    assert len(values) == 15 and all(math.isfinite(v) and v>0 for v in values)
    assert math.isclose(statistics.mean(values),timing['latency_ms'],rel_tol=1e-9)
    assert math.isclose(timing['latency_ms']*1000,timing['profile_kernel_us'],rel_tol=1e-9)
    row['admitted_latency_ms'] = timing['latency_ms']
    if old['admission_metric'] == 'host_roundtrip_with_device_synchronize_mean15':
        row['admission_metric'] = 'sum_profiled_v2_kernel_record_durations_mean15'
        row.pop('device_sum_quarantined',None)
        changed.append(dict(key=row['key'],old_H_ms=old['admitted_latency_ms'],selected_v2_ms=timing['latency_ms']))
    else:
        assert old['admission_metric'] == 'sum_profiled_single_core_role_kernel_durations_mean15'
        assert row == old
    assert {k:v for k,v in row.items() if k not in ('admitted_latency_ms','admission_metric','device_sum_quarantined')} == {
        k:v for k,v in old.items() if k not in ('admitted_latency_ms','admission_metric','device_sum_quarantined')}
assert len(rows)==5550 and len(changed)==2814
digest=dump(OUT/'BASELINE32_PRIVATE.json',dict(catalog,rows=rows,
        frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
for name in ['HISTORICAL_CONTRACT_ROWS.json','FROZEN_MANIFEST_SNAPSHOTS.json']:
    shutil.copyfile(OLD/name,OUT/name)
new_gate=dict(gate,catalog_sha256=digest,device_sum_admitted=5550,conservative_host_fallback=0,
    metric_policy_id='saved-v2-device-sum-user-20261101',
    metric_policy='User explicitly selects original saved v2 device kernel sums for every historical native baseline; H is never admitted. Future v3 native baseline remeasurement is deferred.',
    metric_policy_authority='2026-11-01 explicit user correction; supersedes prior H-fallback decision.',
    supersedes_catalog_sha256=gate['catalog_sha256'],
    known_measurement_caveat='Original v2 sums may double-count AIC/AIV roles of a mixed task. Bias is deliberately retained by user policy pending future v3 baseline remeasurement; these are not v3-remeasured native results.')
dump(OUT/'BASELINE_ADMISSION.json',new_gate)
proof=dict(status='passed',native_rows=5550,restored_from_H=2814,H_admitted=0,
    unchanged_saved_v2_rows=2736,raw_samples_and_correctness_evidence_unchanged=True,
    old_catalog_sha256=gate['catalog_sha256'],new_catalog_sha256=digest,
    original_all15_v2_sample_means_verified=True,device_runs=0,changed=changed)
dump(OUT/'V2_POLICY_TRANSITION.json',proof)
print(json.dumps({k:v for k,v in proof.items() if k!='changed'},indent=2))
