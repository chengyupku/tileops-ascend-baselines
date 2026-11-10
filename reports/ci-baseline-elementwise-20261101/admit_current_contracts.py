"""Preserve audited history and admit only exact current manifest contracts."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    return sha(path)


def main():
    catalog = read('BASELINE32_PRIVATE.json')
    original_gate = read('BASELINE_ADMISSION.json')
    assert original_gate['status'] == 'passed'
    assert sha(ROOT / 'BASELINE32_PRIVATE.json') == original_gate['catalog_sha256']
    manifest = read('FROZEN_MANIFEST_SNAPSHOTS.json')
    current = {(r['op'], r['case_id']): r for r in manifest['cases']}
    assert len(current) == 2208
    kept, historical = [], []
    for row in catalog['rows']:
        now = current.get(tuple(row['key'][1:]))
        if now == row['case']:
            kept.append(row)
        else:
            historical.append(dict(reason='removed_case' if now is None else 'changed_contract',
                                   current_case=now, original_record=row))
    assert len(kept) == 5550 and len(historical) == 23
    assert {x['original_record']['key'][1] for x in historical} == {
        'CausalConv1dFwdOp', 'CausalConv1dUpdateFwdOp'}
    out = ROOT / 'current-contract-baseline'
    out.mkdir(exist_ok=False)
    digest = dump(out / 'BASELINE32_PRIVATE.json', {
        **catalog, 'frozen_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'rows': kept})
    archive_sha = dump(out / 'HISTORICAL_CONTRACT_ROWS.json', historical)
    manifest_sha = dump(out / 'FROZEN_MANIFEST_SNAPSHOTS.json', manifest)
    metrics = collections.Counter(r['admission_metric'] for r in kept)
    gate = {**original_gate, 'catalog_sha256': digest, 'total_unique_pairs': len(kept),
            'device_sum_admitted': metrics['sum_profiled_single_core_role_kernel_durations_mean15'],
            'conservative_host_fallback': metrics['host_roundtrip_with_device_synchronize_mean15'],
            'historical_contract_pairs': len(historical),
            'historical_contract_sha256': archive_sha,
            'manifest_snapshots_sha256': manifest_sha,
            'manifest_cases': len(current),
            'current_workloads_with_native_baseline': len({tuple(r['key'][1:]) for r in kept}),
            'historical_parent_catalog_sha256': original_gate['catalog_sha256'],
            'current_contract_policy': 'Only exact current workload metadata admitted; changed and removed cases retained privately as historical measurements.'}
    dump(out / 'BASELINE_ADMISSION.json', gate)
    print(json.dumps({k:v for k,v in gate.items() if k not in ('audit_inputs',)}, indent=2))


if __name__ == '__main__':
    main()
