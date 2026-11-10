"""Apply inspected policy resolutions to the saved assessment, not the catalog."""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
W = Path('/root/dyq/workspace')
rows = [json.loads(x) for x in (P / 'PAIRS.jsonl').open()]
composition = json.loads((P / 'COMPOSITION_EVIDENCE.json').read_text())
sources = json.loads((P / 'SOURCE_HASHES.json').read_text())
resolved = {}
for c in composition['checkers']:
    key = (c['library'], c['op'], c['case_id'])
    v = c['verification']
    assert c['status'] == 'verified' and c['inputs_preserved'] is True
    assert not c['provider_source_changed'] and not c['oracle_source_changed']
    assert v['checked_outputs'] == v['total_outputs'] and v['total_outputs'] > 0
    assert not v['unchecked_reason']
    suffixes = ['sgl/moe.py'] if c['op'] != 'FP8LightningIndexerFwdOp' else ['sgl/lightning_logits.py', 'sgl/native_fp32_gemv.py']
    for suffix in suffixes:
        path = W / 'TileOPs/benchmarks/ascend_all_pairs' / suffix
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        pins = {v for p, v in c['source_sha256'].items() if p.endswith(suffix)}
        assert pins == {sha}, (key, suffix, pins, sha)
        sources[str(path)] = {'sha256': sha, 'bytes': path.stat().st_size}
    resolved[key] = {'checker': c['path'], 'sha256': c['sha256'], 'policy': 'Existing complete native operators composed at the public API; source-reviewed current adapter matches the successful historical checker.'}
assert len(resolved) == 26

soft_unknown = set()
for group in json.loads((P / 'SGL_BOUNDARY_RECORDS.json').read_text()):
    for record in group['records']:
        k = tuple(record['key'])
        body = record.get('record', {})
        if k[1] in ('RMSNormFwdOp', 'SiluFwdOp'):
            assert body['key'] == list(k)
            assert body['status'] == 'unknown_original_support_or_provenance'
            soft_unknown.add(k)
assert len(soft_unknown) == 6

checker_checks = []
catalog = W / '.worktrees/TileOPs-baseline-catalog/docs/ascend950/baselines'
contracts = {(c['op'], c['case_id']): c for c in json.loads((catalog / 'cases.json').read_text())['cases']}
for group in json.loads((P / 'CHECKER_READS.json').read_text()):
    assert group.get('error') is None
    for c in group['rows']:
        v = c['verification']
        assert c['status'] == 'verified' and c['inputs_preserved'] is True
        assert not c['provider_source_changed'] and not c['oracle_source_changed']
        assert c['candidate_returned'] is True
        assert v['checked_outputs'] == v['total_outputs'] and v['total_outputs'] > 0
        assert not v['unchecked_reason']
        case = contracts[tuple(c['key'][1:])]
        assert c['case_snapshot']['parameters'] == case['params']
        assert c['case_snapshot']['declared_tensors'] == case['tensors']
        checker_checks.append(c['key'])

for r in rows:
    k = tuple(r['key'])
    if k in resolved:
        r['assessment'] = 'confirmed_policy_restorable'
        r['new_boundary_resolution'] = resolved[k]
    elif k in soft_unknown:
        r['assessment'] = 'restoration_candidate_numeric_evidence_no_recorded_boundary_conflict'
        r['new_boundary_resolution'] = {'policy': 'Old reason is generic missing provenance closure, not a demonstrated prohibited implementation change. Still a candidate, not a completed source audit.'}
    elif 'modified_dependency_receipt' in r['boundary_flags']:
        r['assessment'] = 'modified_dependency_not_cleared_by_two_permissions'

counts = collections.Counter(r['assessment'] for r in rows)
assert len(rows) == len({tuple(r['key']) for r in rows}) == 3619
assert counts == {
    'confirmed_policy_restorable': 26,
    'restoration_candidate_numeric_evidence_no_recorded_boundary_conflict': 3463,
    'modified_dependency_not_cleared_by_two_permissions': 87,
    'additional_receipt_binding_needed': 43,
}
summary = {
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'cohort': 3619,
    'assessment_counts': dict(counts),
    'candidate_plus_confirmed': 3489,
    'candidate_plus_confirmed_percent': 100 * 3489 / 3619,
    'confirmed_restorable_by_operator': dict(collections.Counter(r['key'][1] for r in rows if r['assessment'] == 'confirmed_policy_restorable')),
    'modified_dependency_by_library': dict(collections.Counter(r['key'][0] for r in rows if r['assessment'].startswith('modified_dependency'))),
    'numeric_evidence': {
        'prior_exact_acceptance_index': 2437,
        'live_archived_full_checkers_checked': len(checker_checks),
        'additional_composition_checkers_checked_not_in_index': 8,
        'union_keys_with_acceptance_or_checked_receipt': len({tuple(r['key']) for r in rows if r['numeric_historical_acceptance'] or r['numeric_checker_index']} | set(resolved)),
        'all_1128_checker_contracts_match_frozen_catalog': True,
        'all_1128_checkers_full_outputs_inputs_preserved_no_provider_or_oracle_changes': True,
        'direct_linked_historical_acceptance_keys': sum(bool(r['historical_acceptance_links']) for r in rows if r['numeric_historical_acceptance']),
        'remaining_historical_keys_use_verified_checkpoint_chain': 2437 - sum(bool(r['historical_acceptance_links']) for r in rows if r['numeric_historical_acceptance']),
    },
    'scope_of_conclusion': 'Feasibility assessment. 26 confirmed policy-restorable pairs; 3463 candidates have numerical evidence and no recorded specific boundary conflict, but their original-source provenance has not been individually certified by this assessment. Do not label all 3489 as already audited verified.',
    'permission_scope': ['Optional autotune allowed', 'Composition of existing complete public operator APIs allowed', 'No permission to modify internal kernels, precision, passes, schedules or checker'],
    'not_all_old_holds_were_due_to_autotune_or_composition': True,
    'catalog_changed': False,
    'runtime_changed': False,
    'npu_runs': 0,
    'do_not_restart_completed_integration_or_fleet': True,
}
(P / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
(P / 'PAIRS.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
with (P / 'ASSESSMENT.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['library', 'operator', 'case_id', 'assessment', 'old_reason', 'old_boundary_flags'])
    for r in rows:
        writer.writerow([*r['key'], r['assessment'], r['prior_reason'], ';'.join(r['boundary_flags'])])
(P / 'SOURCE_HASHES.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
