"""Freeze existing disposition evidence; never import providers or run a device."""
import collections
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
OUT = ROOT / '.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}


def read(path, csv_file=False):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCES[path] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return list(csv.DictReader(raw.decode('utf-8-sig').splitlines())) if csv_file else json.loads(raw)


def compact(value):
    # These are archive locators relative to the original workspace, not checkout links.
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '')
    if isinstance(value, dict):
        return {compact(k): compact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def write(name, value):
    (OUT / name).write_text(json.dumps(compact(value), ensure_ascii=False, indent=2) + '\n')


inventory = read('.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/inventory.json')
for name, expected in inventory['manifest_sha256'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
write('cases.json', {k: inventory[k] for k in ['schema', 'libraries', 'cases', 'manifest_sha256']})
rows = {}
for row in read('reports/baseline-reconciled-20261026/ALL_PAIRS.csv', True):
    key = tuple(row[k] for k in ('baseline', 'operator', 'case_id'))
    assert key not in rows
    rows[key] = {'key': list(key), **{k: row[k] for k in ['status', 'basis', 'reason', 'evidence']}}
for decision in read('reports/baseline-recheck-608-20261026/ALL611_DECISIONS.csv', True):
    key = tuple(decision[k] for k in ('baseline', 'operator', 'case_id'))
    assert rows[key]['status'] == 'candidate_to_confirm'
    assert decision['decision'] in ('要接', '不要接')
    rows[key].update(status='definite_pending' if decision['decision'] == '要接' else 'excluded',
                     basis='Exact-case reviewed classification, 2026-10-26',
                     reason=decision['reason'], evidence=decision['evidence'])
scope700 = {key for key, row in rows.items() if row['status'] == 'definite_pending'}
assert len(scope700) == 700
accepted = read('reports/baseline-integration-700-20261026/ACCEPTED.json')
seen = set()
for receipt in accepted['rows']:
    key = tuple(receipt['key'])
    assert receipt.get('accepted') and key in scope700 and key not in seen
    seen.add(key)
    rows[key].update(status='completed_original_verified',
                     basis='Original full-checker and registration acceptance ledger',
                     reason=receipt.get('reason', 'Accepted original implementation; no performance claim.'),
                     evidence=receipt.get('checker'), receipt=receipt,
                     host=receipt.get('host', 'recorded_in_receipt'))

manual_files = [('local', 'NON_TLA_STATUS.json'), ('local', 'TLA28_FINAL.json'),
                ('server2', 'PAIR_STATUS.json'), ('server1', 'RESULT.json')]
manual_keys, additions, duplicate_successes, earlier_blocks = set(), [], [], []
for host, filename in manual_files:
    path = f'reports/baseline-single-agent-{host}-20261027/{filename}'
    data = read(path)
    for row in data if isinstance(data, list) else data['rows']:
        key = tuple(row['key'])
        assert key in rows
        manual_keys.add(key)
        success = row.get('accepted') or (row.get('registered') and
                  (row.get('verified') or row.get('status') == 'verified'))
        if not success:
            if rows[key]['status'] == 'completed_original_verified':
                earlier_blocks.append({'key': list(key), 'older_disposition_source': path,
                                       'older_status': row['status'],
                                       'resolution': 'Retain later accepted checker evidence; no downgrade from stale blocker.'})
            else:
                rows[key].update(last_disposition=row['status'], host=host,
                                 reason=row.get('reason') or row.get('finding') or row['status'],
                                 evidence={'final_report': path, 'details': row})
            continue
        if key in seen:
            duplicate_successes.append(list(key))
            continue
        checker = read(str(row['checker']).removeprefix(str(ROOT) + '/'))
        verification = checker['verification']
        assert checker['status'] == 'verified' and checker['inputs_preserved']
        assert (checker['library'], checker['op'], checker['case_id']) == key
        assert verification['checked_outputs'] == verification['total_outputs'] > 0
        assert not verification.get('unchecked_reason')
        assert not checker.get('oracle_source_changed') and not checker.get('provider_source_changed')
        assert row.get('registered')
        registration = read(str(row['registration']).removeprefix(str(ROOT) + '/')) if host == 'local' else None
        if registration:
            assert Path(registration['provider']).is_file()
        if host == 'server2':
            proof = read(row['native_log_proof'])
            assert proof and any('Launch kernel' in item['text'] for item in proof)
            pub = read(str(row['publication'][0]).removeprefix(str(ROOT) + '/'))
            assert hashlib.sha256(Path(pub['provider']).read_bytes()).hexdigest() == pub['sha256']
        else:
            guard = read(str(row['guard']).removeprefix(str(ROOT) + '/'))
            assert guard['returncode'] == 0
        rows[key].update(status='completed_original_verified', host=host,
                         basis='Final single-agent full-checker and published exact registration',
                         reason=row.get('limitation', 'Full checker verified; correctness only.'),
                         evidence={'final_report': path, 'checker': row['checker'],
                                   'registration': row['registration']},
                         receipt={k: v for k, v in row.items() if k != 'snapshot'})
        seen.add(key)
        additions.append(list(key))

status_map = {'completed_original_verified': 'verified', 'completed_retained': 'historical_retained',
              'excluded': 'excluded', 'definite_pending': 'closed_not_integrated'}
for key, row in rows.items():
    row['integration700'] = key in scope700
    row['status'] = status_map[row['status']]
    if row['status'] == 'closed_not_integrated':
        row['closure'] = '2026-10-27 user ended integration; preserve failure/blocker, do not claim success.'
        if key not in manual_keys:
            row['last_disposition'] = 'closed_without_new_single_agent_review'
            row['reason'] = 'Not accepted in final union; closed by user. Prior classification is not failure proof.'
    row['performance_status'] = 'not_assessed_in_this_catalog'
# Apply explicit later retirements after historical receipts are reconciled.
# Numerical success remains historical evidence, never a reason to revive a route.
retirement_path = OUT / 'route_retirements.json'
if retirement_path.is_file():
    retirements = read(str(retirement_path.relative_to(ROOT)))
    for decision in retirements['records']:
        key = tuple(decision['key'])
        assert key in rows and not rows[key]['integration700']
        previous = dict(rows[key])
        rows[key].update(
            status=decision['status'],
            basis='2026-10-27 user-directed retirement of the identified custom truth/reduction route',
            reason=decision['reason'], evidence='route_retirements.json',
            previous_disposition=previous)

raw = ''.join(json.dumps(compact(rows[k]), ensure_ascii=False, separators=(',', ':')) + '\n'
              for k in sorted(rows)).encode()
(OUT / 'observed_pairs.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))

provider_root = ROOT / 'TileOPs/benchmarks/ascend_all_pairs'
files = sorted(provider_root.rglob('*.py'))
markers = ['.cache/', 'reports/', '/root/', '/opt/']
inventory_runtime = {
    'meaning': 'Static inventory, not a runtime portability or dispatch certification.',
    'python_file_count': len(files),
    'path_marker_counts': {m: sum(m in p.read_text() for p in files) for m in markers},
    'files': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
               'path_markers': [m for m in markers if m in p.read_text()]} for p in files],
}
write('runtime_inventory.json', inventory_runtime)
write('provenance.json', {
    'schema': 'ascend-baseline-catalog/v1', 'snapshot_at': datetime.now(timezone.utc).isoformat(),
    'source_workspace': 'workspace-relative archive locators; original archives are not included in this checkout',
    'sources': SOURCES, 'input_artifacts': {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in ['cases.json', 'observed_pairs.jsonl.gz', 'runtime_inventory.json'] +
                    (['route_retirements.json'] if retirement_path.is_file() else [])},
    'central_accepted_unique': len(accepted['rows']), 'additional_manual_verified': additions,
    'deduplicated_manual_successes': duplicate_successes, 'superseded_blocker_records': earlier_blocks,
    'integration700_total': len(scope700), 'integration700_verified': len(seen),
    'integration700_closed': len(scope700 - seen),
    'closure': 'User closed remaining failures/blockers; no implementation/kernel/compiler/checker changes.',
    'verification': 'Historical accepted evidence reused. Four supplemental checker/publication records inspected CPU-only. No new NPU runs.',
    'limitations': ['Historical-retained entries are not newly verified.',
                   'Unobserved Cartesian entries are not proof of unsupported APIs.',
                   'Global correctness coverage is not fresh local install readiness or a performance ranking.',
                   'Source paths and report hashes identify the private evidence archive, not portable runtime dependencies.'],
})
print(json.dumps({'counts': dict(collections.Counter(r['status'] for r in rows.values())),
                  'additional_manual': additions, 'integration700_verified': len(seen)}, ensure_ascii=False))
