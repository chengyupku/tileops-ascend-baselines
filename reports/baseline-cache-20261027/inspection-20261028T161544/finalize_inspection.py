import datetime
import fcntl
import json
import os
import pathlib
import time

ROOT = pathlib.Path('/root/dyq/workspace')
B = ROOT / 'reports/baseline-cache-20261027'
D = B / 'inspection-20261028T161544'

def read(p):
    return json.loads(p.read_text())

def save(p, x):
    q = p.with_name(p.name + '.inspection-tmp')
    q.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')
    os.replace(q, p)

now = time.time()
utc = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat()
next_at = '2026-10-28T20:15:44.214085+00:00'
hosts = ['local', 'server1', 'server2']
before = {h: read(D / (h + '.json'))['summary'] for h in hosts}
after = {h: read(D / (h + '-after.json'))['summary'] for h in hosts}
audits = {}
for h in hosts:
    names = [h + '-AUDIT-final.json'] if h == 'local' else [h + '-AUDIT.json', h + '-AUDIT-delta.json']
    audits[h] = {
        'paths': [str(D / n) for n in names],
        'checked': sum(read(D / n)['checked'] for n in names),
        'usable': sum(read(D / n)['usable'] for n in names),
        'rejected': sum(read(D / n)['rejected'] for n in names),
        'fresh_hashes': all(read(D / n)['fresh_hashes'] for n in names),
        'device_runs': 0,
    }
assert all(a['rejected'] == 0 for a in audits.values()), 'Do not finalize healthy audit with rejection'
effect = read(D / 'ACTION_EFFECT.json')
effect.pop('current_live', None)
result = {
    'kind': 'four_hour_three_host_cache_inspection',
    'due_at_utc': '2026-10-28T16:15:44.214085+00:00',
    'checked_at': now, 'checked_at_utc': utc,
    'report': str(D / 'RESULT.json'), 'assignment_revision': 6, 'scope_count': 6419,
    'plan_sha256': after['local']['plan_sha256'],
    'before': before, 'after': after, 'audits': audits,
    'audit_usable': sum(a['usable'] for a in audits.values()),
    'audit_rejected': 0,
    'accepted_last_hour': sum(a['windows']['1']['cached'] for a in after.values()),
    'accepted_last_four_hours': sum(a['windows']['4']['cached'] for a in after.values()),
    'timing_caveats': [
        'Remote clocks approximately 93 seconds behind local; rolling windows use each host clock.',
        'Two utilization samples five seconds apart are not sustained utilization evidence.',
        'Initial server1 child completion was a natural pair boundary, not supervisor death.',
        'Local negative live_age in post-snapshot is a concurrent read race of about two seconds.',
        'Audit prefixes and live snapshots have different cutoffs; cache growth continues.',
    ],
    'ranked_problems': [
        {'rank': 1, 'problem': 'Repeated device startup for CPU-deterministic missing provider or unaccepted exact ops-nn route',
         'affected_remaining_pairs_proven': 359, 'new_exact_holds': 356,
         'observed_prior_hour_attempts': 49, 'observed_attempt_wall_seconds': 743,
         'lost_card_minutes': None, 'confidence': 'high; isolated CPU import/factory probe plus actual failure logs',
         'expected_accepted_throughput_effect': 'Avoid known no-output startups; not yet quantified as an accepted-rate gain',
         'cost_risk': 'Low: reversible exact-pair holds only; no eligibility or implementation change',
         'evidence': ['ROUTE_PROBE.json', 'ADMISSION_HOLDS.json', 'local-error-ranking.json']},
        {'rank': 2, 'problem': 'Existing exact FFT/BF16 routes lack coherent original startup environment',
         'affected_routes': 37, 'historical_failures_eligible_for_changed_condition_retry': 25,
         'lost_card_minutes': None, 'expected_accepted_throughput_effect': None,
         'confidence': 'high for startup and frozen-source causes; no demonstrated cache gain',
         'cost_risk': 'Scoped CPU launcher/recipe selection; full unchanged admission/checker/profiler gates retained',
         'evidence': ['ENV_RETRY_CHANGES.json', 'FFT_MEASUREMENT_HOLD.json', 'BF16_FROZEN_HOLD.json']},
        {'rank': 3, 'problem': 'Remaining numerical/compiler/contract failures and narrower guard-context startup issues',
         'lost_card_minutes': None, 'expected_accepted_throughput_effect': None,
         'action': 'Preserve exact failure history; no unchanged retry or whole-library extrapolation'},
    ],
    'actions': [
        {'action': 'Added 356 exact admission holds under QUARANTINE.lock', 'effect': effect},
        {'action': 'Selected original frozen recipes before guarded imports, restored OPP after site hooks, normalized enabled default autoload representation',
         'files': ['tools/baseline_cache_python.sh', 'tools/baseline_cache_exact_env_v1.py', 'tools/baseline_cache_exact_env_v1.json'],
         'scope': 'local exact 37 keys only; original source and native pins retained',
         'verified_new_usable_caches_attributable_to_this_fix': 0},
        {'action': 'Held 10 exact FFT routes after two full-correctness passes emitted no required FFTS profiler records',
         'evidence': 'FFT_MEASUREMENT_HOLD.json', 'no_timing_protocol_fallback': True},
        {'action': 'Held 27 exact BF16 routes: four original manifests pin a different fla/__init__.py hash',
         'evidence': 'BF16_FROZEN_HOLD.json', 'pins_or_source_not_rewritten': True},
        {'action': 'Restored unlimited local CPU supervisor at normal pair boundary; verified one new normal cache after resume',
         'evidence': ['LOCAL_RESUMED.json', 'ACTION_EFFECT.json']},
    ],
    'new_exact_holds': 393,
    'holds_are_not': ['verified success', 'proof of baseline algorithm defect', 'whole-library rejection'],
    'unchanged_boundaries': ['original kernels/compiler/schedule/checker', 'full correctness and all outputs/input protection',
        '3 warmups/15 profiler measurements/1GiB flush/no Events', 'exclusive host guard', 'no valid cache reruns',
        'no foreign task termination', 'current unique PLAN ownership'],
    'followup': [
        'Continue all three existing CPU supervisors; next idle-only inspection at original 20:15 UTC anchor.',
        'Check actual consumption/output after 356 admission holds; do not claim a measured rate improvement yet.',
        'BF16 requires coherent original frozen deployment or separately reviewed top-level revalidation; never blindly refresh pins.',
        'FFT requires original native profiler coverage; no Events or relaxed protocol.',
        'Server1 free disk approximately 5.37 GiB: monitor growth; do not remove foreign files.',
        'Retain server1 DeepGEMM47 dependency hold and prior exact holds until actual conditions change.',
    ],
    'next_check': next_at,
}
assert not (D / 'RESULT.json').exists()
save(D / 'RESULT.json', result)
entry = {k: result[k] for k in ['kind', 'checked_at', 'checked_at_utc', 'report', 'assignment_revision', 'scope_count',
    'audit_usable', 'audit_rejected', 'accepted_last_hour', 'accepted_last_four_hours', 'new_exact_holds', 'next_check']}
entry['audits'] = audits
entry['action_evidence'] = str(D / 'ACTION_EFFECT.json')
with (B / 'INSPECTIONS.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (B / 'INSPECTIONS.jsonl').open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    save(B / 'LAST_INSPECTION.json', entry)
with (B / 'RECURRING_ALARMS.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    p = B / 'RECURRING_ALARMS.json'; a = read(p)
    matches = [m for m in a['milestones'] if abs(m['due_at'] - 1793204144.2140849) < 0.01]
    assert len(matches) == 1
    matches[0].update(checked_at=now, inspection_report=str(D / 'RESULT.json'))
    assert any(abs(m['due_at'] - 1793218544.2140849) < 0.01 for m in a['milestones'])
    a['message'] = a['message'].replace('下次16:15UTC', '下次20:15UTC')
    a['message'] += '\n16:15巡检已完成：inspection-20261028T161544/RESULT.json。新增local精确暂挂393项（356 provider/准入、10 FFT无FFTS记录、27 BF16原冻结源码哈希不符），非数值失败或成功计数。已有guard/checker/源哈希不放宽。local仅在pair边界恢复；三端继续产出。CPU启动选择器tools/baseline_cache_exact_env_v1.py/json只针对37个原生配方，不要重写原冻结pin；其环境修复尚无新增合格缓存。调试前先CPU检查完整冻结依赖闭包，避免逐层设备启动失败。以当前PLAN/日志为准；不得重跑有效缓存。\n'
    save(p, a)
state_path = ROOT / '.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with state_path.with_name('.state.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = read(state_path)
    cache = state['baseline_cache_20261027']
    save(D / 'MEMORY_before_baseline_cache.json', cache)
    cache.update(updated_at=utc, status='three_host_cache_sweep_running', last_inspection=entry,
        recurring_next_due_at=next_at,
        current_ready={h: after[h]['ready'] for h in hosts},
        current_cached={h: after[h]['cached'] for h in hosts},
        current_held={h: after[h]['held'] for h in hosts},
        current_counts_observed_at={h: after[h]['host_time'] for h in hosts})
    cache['active_followup'] = 'Three native CPU supervisors continue serial guarded pairs. Fresh audits pass; local exact holds avoid known startup failures. Next20:15UTC idle-only4h anchor; do not resurrect fleets. Use RESULT report for all boundaries and follow-up.'
    cache['last_cache_repair'] = {
        'report': str(D / 'RESULT.json'), 'new_exact_holds': 393,
        'startup_selector': 'tools/baseline_cache_exact_env_v1.py + .json',
        'verified_new_caches_attributable_to_startup_fix': 0,
        'production_cache_growth_after_resume': 1,
        'frozen_dependency_lesson': 'Before guarded pilots, CPU-check the entire original recipe/environment and frozen-pin dependency closure. Initial pilots exposed startup gates sequentially; preflight closure would avoid that wasted device startup. Do not relax pins or checker to manufacture success.',
        'release_conditions': '356 current public route unavailable/admission missing; FFT10 need original FFTS coverage; BF16 27 need coherent original frozen deployment or reviewed interface-only revalidation. Record real condition changes via RETRY revision; preserve old failures.',
    }
    save(state_path, state)
print(json.dumps(entry, ensure_ascii=False, indent=2))
