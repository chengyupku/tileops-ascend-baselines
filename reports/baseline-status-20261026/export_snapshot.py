"""Read-only ledger export. Does not approve, run, or reclassify a baseline."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

W = Path('/root/dyq/workspace')
OUT = Path(__file__).resolve().parent
B = W / '.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557'
SOURCE = B / 'reconciled_scope_inspection0240'
MODULES = {'DeepEP-Ascend': 'deepep', 'DeepGEMM-Ascend': 'deepgemm',
           'DeepSelect': 'deepselect', 'FlashMLA': 'flashmla', 'TileKernels': 'tilekernels',
           'flash-linear-attention-npu': 'fla', 'sgl-kernel-npu': 'sgl',
           'tilelang-ascend': 'tla', 'vllm-ascend': 'vllm', 'atvc': 'atvc',
           'catlass': 'catlass', 'tilelang': 'tilelang', 'ops-math': 'ops_math',
           'ops-nn': 'ops_nn', 'ops-transformer': 'ops_transformer',
           'ops-tensor': 'ops_tensor', 'ops-fft': 'ops_fft'}
LABELS = {'retained_pass': '接好了且保留', 'excluded_old_success': '曾报成功，现已撤销资格并移出目标',
          'retained_not_passed': '尚无合规成功，目标保留', 'excluded_no_old_success': '未报成功，且确认不用接',
          'unresolved': '资格或历史来源待确认'}


def category(r):
    c = r['classification']
    if c == 'original_supported_checked':
        return 'retained_pass'
    if c == 'proven_excluded_non_target':
        return 'excluded_old_success' if r.get('historical_success_claim') else 'excluded_no_old_success'
    if c == 'eligible_remaining_failed_deferred':
        return 'retained_not_passed'
    assert c == 'unknown_classification'
    return 'unresolved'


def paths(v):
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(k, str) and k.startswith(('/root/', '/opt/', '/tmp/', '/var/', '/usr/')):
                yield k
            yield from paths(x)
    elif isinstance(v, list):
        for x in v:
            yield from paths(x)
    elif isinstance(v, str) and v.startswith(('/root/', '/opt/', '/tmp/', '/var/', '/usr/')):
        yield v


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csv_write(name, rows, fields):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


raw = (SOURCE / 'EXACT_TARGET_SETS.jsonl').read_bytes()
rows = [json.loads(s) for s in raw.splitlines()]
assert len({tuple(r['key']) for r in rows}) == len(rows)
source_summary = json.loads((SOURCE / 'SUMMARY.json').read_text())
assert dict(collections.Counter(r['classification'] for r in rows)) == source_summary['counts']
(OUT / 'EXACT_TARGET_SETS.snapshot.jsonl').write_bytes(raw)
dump('SOURCE_SUMMARY.snapshot.json', source_summary)

# These are already published joins, not new worker acceptance or scientific review.
supplement = collections.defaultdict(list)
for p in [B / 'inspection_0214/DELTA_OWNER_RECORDS.json', B / 'inspection_0114/DELTA_OWNER_RECORDS.json']:
    if p.exists():
        data = json.loads(p.read_text())
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict) and isinstance(r.get('key'), list):
                    supplement[tuple(r['key'])].append({'source': str(p), 'record': r})

exports, indexes, missing_owner = [], [], []
by_library = collections.defaultdict(collections.Counter)
for lineno, r in enumerate(rows, 1):
    key = tuple(r['key'])
    cat = category(r)
    by_library[key[0]][cat] += 1
    refs = sorted(set(paths(r)))
    qualified_refs = sorted(set(paths({k: v for k, v in r.items() if k not in
                                      ('evidence', 'scope1529_evidence', 'classification_history', 'original_assignment')})))
    extra = supplement.get(key, [])
    qualified_refs = sorted(set(qualified_refs + list(paths(extra))))
    host = (r.get('original_assignment') or {}).get('host') or ''
    if not host:
        for candidate in ('local', 'server1', 'server2'):
            if any('/' + candidate + '/' in x or ('local24' in x and candidate == 'local') for x in qualified_refs):
                host = candidate
                break
    module = MODULES.get(key[0], '')
    shared = str(W / 'TileOPs/benchmarks/ascend_all_pairs' / module) if module else ''
    x = {'category': cat, 'category_zh': LABELS[cat], 'baseline': key[0], 'operator': key[1],
         'case_id': key[2], 'family': r.get('family', ''), 'historical_success_claim': r.get('historical_success_claim', False),
         'classification': r['classification'], 'numerator': r.get('numerator'), 'denominator': r.get('denominator'),
         'owner_host_hint': host, 'shared_provider_directory_local': shared,
         'snapshot_line': lineno, 'reason_tag': r.get('substatus'),
         'qualified_evidence_references': json.dumps(qualified_refs, ensure_ascii=False),
         'all_evidence_references': json.dumps(refs, ensure_ascii=False)}
    exports.append(x)
    if cat == 'retained_pass':
        # Location references include legacy evidence. Their presence alone does not
        # certify that every file in a provider directory is retained or current.
        indexes.append({'key': r['key'], 'snapshot_line': lineno, 'owner_host_hint': host,
                        'shared_provider_directory_local': shared,
                        'qualified_evidence_references': qualified_refs,
                        'all_evidence_references': refs, 'joined_owner_records': extra,
                        'qualification_record': r})
        if not qualified_refs:
            missing_owner.append(r['key'])

fields = list(exports[0])
csv_write('ALL_CASES.csv', exports, fields)
for cat in LABELS:
    csv_write(cat + '.csv', [r for r in exports if r['category'] == cat], fields)
with (OUT / 'RETAINED_CODE_AND_EVIDENCE.jsonl').open('w') as f:
    for r in indexes:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
library_rows = [{'baseline': lib, **{cat: count[cat] for cat in LABELS}, 'total': sum(count.values())}
                for lib, count in sorted(by_library.items())]
csv_write('BY_BASELINE.csv', library_rows, ['baseline', *LABELS, 'total'])
summary = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'snapshot_at': source_summary['at'], 'source': str(SOURCE),
           'source_sha256': hashlib.sha256(raw).hexdigest(),
           'unit': '(baseline library, operator, case_id); not unique operators or Python files',
           'observed_candidate_union': len(rows), 'global_denominator': None,
           'counts': dict(collections.Counter(r['category'] for r in exports)),
           'retained_with_historical_success_claim': sum(r['category'] == 'retained_pass' and r['historical_success_claim'] for r in exports),
           'retained_not_passed_with_old_success_claim': sum(r['category'] == 'retained_not_passed' and r['historical_success_claim'] for r in exports),
           'unresolved_with_old_success_claim': sum(r['category'] == 'unresolved' and r['historical_success_claim'] for r in exports),
           'retained_missing_qualified_path_refs': missing_owner,
           'warning': 'Old success claims are historical ledger labels, not proof of old compliance. Exclusion is not a count of physically reverted files.',
           'by_baseline': library_rows}
dump('SUMMARY.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'by_baseline'}, ensure_ascii=False))
