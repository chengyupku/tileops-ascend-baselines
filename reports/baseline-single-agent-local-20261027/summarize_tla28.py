"""Reconcile exact-case evidence without importing a device library."""
import collections
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE/'TLA28_PLAN.json').read_text())
history = []
for name in ('TLA28_RESULTS.json', 'TLA28_RETRY_RESULTS.json'):
    f = HERE/name
    if f.exists():
        history.extend(json.loads(f.read_text()))
latest = {tuple(r['key']): r for r in history}
rows = []
for item in plan:
    row = dict(latest.get(tuple(item['key']), item))
    row['verified'] = False
    row['registered'] = False
    out = HERE/row['attempt_id']
    row['native_launched_in_this_attempt'] = (out/'ACTUAL_LAUNCH.json').exists()
    if row.get('action') == 'retain_existing_hang_skip':
        row['final_status'] = 'prior_exact_timeout_skipped'
        prior = json.loads(Path(row['evidence']).read_text())
        row['prior_status'] = prior['status']
        row['prior_conflict_detected'] = prior.get('conflict_detected', prior.get('outside_interference'))
    elif row.get('action') == 'original_api_contract_unsupported':
        row['final_status'] = 'original_api_contract_unsupported'
    elif (out/'checker.json').exists():
        check = json.loads((out/'checker.json').read_text())
        v = check.get('verification') or {}
        complete = (check.get('status') == 'verified' and check.get('inputs_preserved') is True
                    and v.get('checked_outputs', 0) > 0
                    and v.get('checked_outputs') == v.get('total_outputs'))
        row['verified'] = complete
        row['verification'] = v
        row['final_status'] = ('verified_pending_registration' if complete
                               else 'checker_' + check.get('stage', 'unknown') + '_' + check.get('status', 'unknown'))
    else:
        row['final_status'] = row.get('status', 'not_finished')
    receipt = out/'guard.json'
    if receipt.exists():
        g = json.loads(receipt.read_text())
        row['host'] = g.get('host', 'local')
        row['guard_timeout_seconds'] = int(g['command'][g['command'].index('--timeout')+1])
        assert row['guard_timeout_seconds'] <= 120
    rows.append(row)
assert len(rows) == len({tuple(r['key']) for r in rows}) == 28
final_statuses = {'prior_exact_timeout_skipped', 'original_api_contract_unsupported',
                  'device_sync_timeout_skip', 'guarded_command_timeout_skip',
                  'checker_build_failed', 'checker_run_failed', 'checker_verify_failed',
                  'verified_pending_registration'}
pending = [r['key'] for r in rows if r['final_status'] not in final_statuses]
by_operator = {}
for row in rows:
    counts = by_operator.setdefault(row['key'][1], {})
    status = row['final_status']
    counts[status] = counts.get(status, 0) + 1
result = {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'28 exact tilelang-ascend cases only', 'case_count':28,
          'all_cases_dispositioned':not pending, 'pending_cases':pending,
          'verified_cases':sum(r['verified'] for r in rows),
          'status_counts':dict(collections.Counter(r['final_status'] for r in rows)),
          'status_counts_by_operator':by_operator,
          'new_guarded_attempts_including_setup_failures':sum(bool(r.get('newly_executed')) for r in history),
          'one_agent':True, 'delegation':False, 'baseline_kernel_changes':False,
          'compiler_changes':False, 'checker_changes':False, 'performance_measured':False,
          'new_runtime_timeout_seconds':120,
          'retained_prior_hangs_not_rerun':4,
          'rows':rows}
(HERE/'TLA28_FINAL.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('case_count','verified_cases','status_counts')}))
