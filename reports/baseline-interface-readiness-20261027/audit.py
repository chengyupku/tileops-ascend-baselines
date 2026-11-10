"""CPU-only, timestamped readiness join; never executes providers or device code."""
import collections
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
OUT = Path(__file__).resolve().parent
PATHS = {
    'inventory': ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json',
    'base': ROOT / 'reports/baseline-reconciled-20261026/ALL_PAIRS.csv',
    'decisions': ROOT / 'reports/baseline-recheck-608-20261026/ALL611_DECISIONS.csv',
    'accepted': ROOT / 'reports/baseline-integration-700-20261026/ACCEPTED.json',
    'previous_confirmed': ROOT / 'reports/baseline-interface-readiness-20261026/confirmed_operators.csv',
    'previous_all': ROOT / 'reports/baseline-interface-readiness-20261026/operators.csv',
}
RAW = {k: p.read_bytes() for k, p in PATHS.items()}


def csv_rows(name):
    return list(csv.DictReader(io.StringIO(RAW[name].decode('utf-8-sig'))))


def key(row):
    return tuple(row[x] for x in ('baseline', 'operator', 'case_id'))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


inventory = json.loads(RAW['inventory'])
for name, sha in inventory['manifest_sha256'].items():
    path = ROOT / name
    if not path.exists():
        path = ROOT / 'TileOPs' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, path
cases = collections.defaultdict(set)
for case in inventory['cases']:
    cases[case['op']].add(case['case_id'])
assert len(cases) == 205

base = csv_rows('base')
rows = {key(r): dict(r) for r in base}
assert len(rows) == len(base) == 10833
for decision in csv_rows('decisions'):
    k = key(decision)
    assert rows[k]['status'] == 'candidate_to_confirm'
    assert decision['decision'] in ('要接', '不要接')
    rows[k].update(
        status='definite_pending' if decision['decision'] == '要接' else 'excluded',
        reason=decision['reason'], evidence=decision['evidence'],
        basis='Reviewed exact-case classification, 2026-10-26',
    )
before = {k: dict(r) for k, r in rows.items()}
accepted = json.loads(RAW['accepted'])
accepted_by_key = {}
for receipt in accepted['rows']:
    if not receipt.get('accepted'):
        continue
    k = tuple(receipt['key'])
    assert k not in accepted_by_key
    assert k in rows and rows[k]['status'] == 'definite_pending', k
    accepted_by_key[k] = receipt
    rows[k].update(status='completed_original_verified',
                   basis='integration700 parent-accepted original implementation',
                   reason='Full canonical correctness and formal registration accepted; not performance acceptance.',
                   evidence=json.dumps(receipt.get('checker'), ensure_ascii=False))


def summarize(pair_rows):
    by_op = collections.defaultdict(list)
    for row in pair_rows.values():
        by_op[row['operator']].append(row)
    result = []
    for op, ids in sorted(cases.items()):
        rr = by_op[op]
        done = [r for r in rr if r['status'].startswith('completed')]
        pending = [r for r in rr if r['status'] not in
                   ('completed_original_verified', 'completed_retained', 'excluded')]
        missing = sorted(ids - {r['case_id'] for r in done})
        retained = sum(r['status'] == 'completed_retained' for r in done)
        ready = not pending and not missing
        result.append({
            'op': op, 'case_count': len(ids), 'ready_including_user_retained': ready,
            'ready_without_retained_success': ready and retained == 0,
            'completed_pairs': len(done), 'retained_pairs': retained,
            'pending_pairs': len(pending),
            'missing_case_baselines': ';'.join(missing),
            'applicable_baselines': json.dumps(dict(sorted(collections.Counter(
                r['baseline'] for r in done).items())), ensure_ascii=False),
        })
    return result


current = summarize(rows)
previous = summarize(before)
before_ready = {r['op'] for r in previous if r['ready_including_user_retained']}
old_confirmed = {r['op'] for r in csv_rows('previous_confirmed')}
ready = [r for r in current if r['ready_including_user_retained']]
strict = [r for r in ready if r['ready_without_retained_success']]
new_strict = [r for r in strict if r['op'] not in old_confirmed]
new_ready = [r for r in ready if r['op'] not in before_ready]
assert old_confirmed <= {r['op'] for r in strict}
old_all = {r['op']: r for r in csv_rows('previous_all')}
for row in new_strict:
    assert old_all[row['op']]['old_nonunsupported_claims_outside_current_ledger'] == '0'

for name, data in [('operators.csv', current), ('ready_including_retained.csv', ready),
                   ('ready_without_retained.csv', strict), ('new_without_retained.csv', new_strict),
                   ('newly_completed_integration700.csv', new_ready)]:
    write_csv(name, data)
write_csv('pairs.csv', list(rows.values()))
new_ops = {r['op'] for r in new_strict}
with (OUT / 'new_operator_evidence.jsonl').open('w') as f:
    for k, row in sorted(rows.items()):
        if k[1] in new_ops:
            f.write(json.dumps({'key': k, 'row': row,
                                'new_acceptance': accepted_by_key.get(k)}, ensure_ascii=False) + '\n')

summary = {
    'audited_at': datetime.now(timezone.utc).isoformat(),
    'accepted_snapshot_updated_at': accepted.get('updated_at'),
    'accepted_new_pairs': len(accepted_by_key),
    'operators': len(current), 'manifest_cases': sum(map(len, cases.values())),
    'manifest_files_hash_checked': len(inventory['manifest_sha256']),
    'observed_pair_counts': dict(collections.Counter(r['status'] for r in rows.values())),
    'previous_conservative_list_count': len(old_confirmed),
    'ready_without_retained_success': len(strict), 'new_without_retained_success': new_strict,
    'ready_including_retained_success': len(ready),
    'ready_before_integration700_including_retained': len(before_ready),
    'newly_completed_integration700_count': len(new_ready),
    'newly_completed_integration700': [r['op'] for r in new_ready],
    'gemm_pending_by_baseline': dict(collections.Counter(r['baseline'] for r in rows.values()
        if r['operator'] == 'GemmFwdOp' and r['status'] == 'definite_pending')),
    'method': [
        'Exact (baseline, operator, case_id) join of reconciled ledger, reviewed classifications and parent acceptance; no worker-only success credited.',
        'Every manifest case has at least one completed baseline; no identified applicable pair remains pending.',
        'Explicit exclusions apply per case. A library may support only a subset, e.g. BiasGateGelu supports the FP16 case, not the two BF16 cases.',
        'No-retained view does not use completed_retained as proof. Broader view honors the existing user-retained-success decision.',
        'New3 compares with the earlier conservative9 list. New29 compares before/after integration700 using the same broader predicate.',
    ],
    'new_operator_explanation': {
        'GeluTanhAndMulFwdOp': 'ops-nn 3 cases already verified; ops-transformer FP16 now accepted; two BF16 cases excluded for original hard-coded half dtype. Total4 applicable pairs.',
        'TanhFwdOp': 'ops-math5 and ops-nn5 already verified; the remaining TLA5 candidates excluded for original exp-based overflow on legal inputs.',
        'MoEPermuteAlignFwdOp': 'DeepEP12 already original-qualified; remaining SGL12 candidates excluded because original whole APIs do not produce per-expert padding and all3 outputs.',
    },
    'performance': 'Readiness concerns interface/correctness evidence. Current integration700 explicitly does not require benchmarking. GELU new checker profile_requested=false. No new all-baseline fastest-per-case performance certification asserted.',
    'runtime_location': 'This is cross-host readiness, not a fresh local install audit. New GELU formal module is present on server1; DeepEP original receipts are on server2.',
    'limitations': [
        '10833 observed pairs are not the complete35054 Cartesian product. Statements apply to currently identified applicable pairs.',
        'Historical accepted original evidence is reused; no full new numerical replay or portability audit.',
        'The earlier nine-op selection also required every qualifying library to cover every case; here explicitly excluded library-case pairs are allowed, matching per-pair applicability.',
        'Running integration may change counts after this frozen acceptance snapshot.',
    ],
    'sources': {k: {'path': str(p), 'sha256': hashlib.sha256(RAW[k]).hexdigest()} for k, p in PATHS.items()},
    'new_npu_runs': 0,
}
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ['audited_at', 'accepted_snapshot_updated_at',
    'ready_without_retained_success', 'ready_including_retained_success',
    'newly_completed_integration700_count', 'gemm_pending_by_baseline']}, ensure_ascii=False))
