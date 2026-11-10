"""CPU-only operator coverage of frozen, audited performance-cache prefixes."""
from collections import Counter, defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / 'reports/baseline-cache-20261027'
CATALOG = ROOT / '.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines'
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


inventory = read(CATALOG / 'cases.json')
plan = read(CAMPAIGN / 'PLAN.json')
manifest_errors = []
for name, digest in inventory['manifest_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        manifest_errors.append(name)
assert not manifest_errors, manifest_errors

cases = defaultdict(set)
for row in inventory['cases']:
    cases[row['op']].add(row['case_id'])
planned = defaultdict(set)
hosts = {}
for row in plan['pairs']:
    key = tuple(row['key'])
    assert key[2] in cases[key[1]]
    planned[key[1]].add(key)
    hosts[key] = row['host']

raw_catalog = (CATALOG / 'observed_pairs.jsonl.gz').read_bytes()
sources[str((CATALOG / 'observed_pairs.jsonl.gz').relative_to(ROOT))] = hashlib.sha256(raw_catalog).hexdigest()
recognized = defaultdict(set)
unresolved = defaultdict(set)
for line in gzip.decompress(raw_catalog).decode().splitlines():
    row = json.loads(line)
    key = tuple(row['key'])
    if row['status'] in ('verified', 'historical_retained'):
        recognized[key[1]].add(key)
    elif row['status'] != 'excluded':
        unresolved[key[1]].add(key)


def analyze(name, audits):
    cached = set()
    for audit in audits:
        assert audit['rejected'] == 0
        for row in audit['rows']:
            assert row['usable'] and not row['errors']
            key = tuple(row['key'])
            assert hosts[key] == audit['host']
            assert key not in cached
            cached.add(key)
    records = []
    for op in sorted(cases):
        wanted = planned[op]
        actual = {key for key in cached if key[1] == op}
        covered_cases = {key[2] for key in actual}
        missing = wanted - cached
        outside_plan = recognized[op] - wanted
        ready = (bool(wanted) and not missing and covered_cases == cases[op]
                 and not outside_plan and not unresolved[op])
        records.append({
            'op': op, 'manifest_cases': len(cases[op]),
            'cases_with_cache': len(covered_cases),
            'planned_pairs': len(wanted), 'cached_pairs': len(actual),
            'missing_planned_pairs': len(missing),
            'recognized_pairs_outside_plan': len(outside_plan),
            'unresolved_catalog_pairs': len(unresolved[op]),
            'all_cases_all_registered_baselines_cached': ready,
            'libraries': dict(sorted(Counter(key[0] for key in wanted).items())),
            'hosts': sorted({hosts[key] for key in actual}),
            'cases_without_cache': sorted(cases[op] - covered_cases),
            'missing_pairs': [list(key) for key in sorted(missing)],
        })
    return {'snapshot': name, 'cached_pairs': len(cached),
            'distinct_cached_cases': len({key[1:] for key in cached}),
            'ready': [row for row in records if row['all_cases_all_registered_baselines_cached']],
            'all_operators': records}


old_summary = read(CAMPAIGN / 'REUSE_AUDIT_1736_20261028.json')
old_audits = []
current_audits = []
for host in ('local', 'server2'):
    old_path = ROOT / old_summary['hosts'][host]['report']
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old_summary['hosts'][host]['report_sha256']
    old_audits.append(read(old_path))
    current_audits.append(read(HERE / f'current-audit-{host}.json'))
old = analyze('previously_reported_1736', old_audits)
current = analyze('current_frozen_prefix', current_audits)
assert old['cached_pairs'] == 1736
summary = {
    'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'snapshot': read(HERE / 'snapshot.json')['snapshot_at'],
    'manifest_operators': len(cases), 'manifest_cases': sum(map(len, cases.values())),
    'manifest_files_hash_verified': len(inventory['manifest_sha256']),
    'planned_pairs': len(plan['pairs']),
    'definition': 'Every manifest case has a usable cache, all planned baseline pairs are cached, and no recognized or unresolved catalog pair lies outside that complete coverage.',
    'previous': {k: v for k, v in old.items() if k != 'all_operators'},
    'current': {k: v for k, v in current.items() if k != 'all_operators'},
    'limits': [
        'Coverage is over registered applicable candidates; this is not an exhaustive search for every possible implementation.',
        'The6419-pair campaign does not cover every manifest case; completing its per-op queue alone is insufficient.',
        'Usability is specific to the original host/software/case/baseline/reference/protocol. No cross-host winner comparison is made.',
        'Current audit uses verified unchanged file stamps where available and hashes changed/unmemoized files; prior1736 audit used fresh hashes.',
        'CPU-only reads/audits; no NPU execution, no campaign or baseline-branch modifications.',
    ],
    'source_sha256': sources,
}
(HERE / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
(HERE / 'operators.json').write_text(json.dumps(current['all_operators'], ensure_ascii=False, indent=2) + '\n')
columns = ['op', 'manifest_cases', 'cases_with_cache', 'planned_pairs', 'cached_pairs',
           'missing_planned_pairs', 'recognized_pairs_outside_plan', 'unresolved_catalog_pairs',
           'all_cases_all_registered_baselines_cached', 'libraries', 'hosts']
with (HERE / 'operators.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in current['all_operators']:
        writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (dict, list)) else row[k]
                         for k in columns})
print(json.dumps({k: summary[k] for k in ['checked_at', 'manifest_operators', 'previous', 'current']}, ensure_ascii=False, indent=2))
