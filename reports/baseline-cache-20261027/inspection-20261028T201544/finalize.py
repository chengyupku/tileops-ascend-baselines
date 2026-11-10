import datetime
import fcntl
import json
import os
import pathlib
import time

ROOT = pathlib.Path('/root/dyq/workspace')
B = ROOT / 'reports/baseline-cache-20261027'
D = B / 'inspection-20261028T201544'
read = lambda p: json.loads(p.read_text())

def save(p, obj):
    tmp = p.with_name(p.name + '.inspection-201544.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, p)

now = time.time()
utc = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat()
next_check = '2026-10-29T00:15:44.214085+00:00'
hosts = ['local', 'server1', 'server2']
before = {h: read(D / (h + '.json'))['summary'] for h in hosts}
after = {h: read(D / (h + '-after.json'))['summary'] for h in hosts}
effects = {h: read(D / (h + '-EFFECT-final2.json')) for h in hosts}
audits = {}
for h in hosts:
    paths = [D / (h + '-AUDIT.json'), D / (h + '-AUDIT-delta.json')]
    a = [read(p) for p in paths]
    audits[h] = {'paths': [str(p) for p in paths],
                 'checked': sum(x['checked'] for x in a),
                 'usable': sum(x['usable'] for x in a),
                 'rejected': sum(x['rejected'] for x in a),
                 'fresh_hashes': all(x['fresh_hashes'] for x in a), 'device_runs': 0}
assert all(x['rejected'] == 0 for x in audits.values())
assert len({s['plan_sha256'] for s in before.values()}) == 1
for h in hosts:
    assert not effects[h]['held_device_attempts_after_action']
holds = {h: read(D / (h + '-FP16_HOLDS.json'))['count'] for h in hosts}
record = {
    'kind': 'four_hour_three_host_cache_inspection',
    'due_at_utc': '2026-10-28T20:15:44.214085+00:00',
    'checked_at': now, 'checked_at_utc': utc,
    'report': str(D / 'RESULT.json'), 'assignment_revision': 6, 'scope_count': 6419,
    'plan_sha256': before['local']['plan_sha256'],
    'before': before, 'after': after, 'effects': effects, 'audits': audits,
    'audit_usable': sum(a['usable'] for a in audits.values()), 'audit_rejected': 0,
    'accepted_last_hour': sum(x['windows']['1']['cached'] for x in before.values()),
    'accepted_last_four_hours': sum(x['windows']['4']['cached'] for x in before.values()),
    'new_exact_holds': holds,
    'all_hosts_native_supervisor_identity_verified': all(effects[h]['native']['supervisor']['matches'] for h in hosts),
    'common_window_findings': [
        'All three native CPU supervisors alive; host guard health checks rc0/value0 and no observer errors.',
        'No resource_wait wall time in sampled rolling1h/4h journals. Hardware-lock queue is not the throughput bottleneck.',
        'Two utilization samples five seconds apart cannot establish sustained occupancy. Do not claim cards are saturated.',
        'server1 transient exited child/zombie guard at measured_pending_guard is a normal pair boundary; next pair and cache growth verified separately.',
        'Remote clocks approximately93 seconds behind local. Per-host rolling windows and timestamp-labelled evidence used.',
        'Local large DSA full correctness may remain active across inspection; do not confuse a long required check with death or cancel its guard.',
    ],
    'ranked_findings': [
        {'rank_by_observed_wall_loss': 1, 'issue': 'Original numerical failures, mainly ops-transformer BF16 GDN',
         'attempts': 67, 'attempt_wall_seconds': 2379, 'lost_card_minutes': None,
         'action': 'Preserve failures and complete untested distinct cases. Changing math/checker or inferring all cases fail is not authorized.',
         'repairability_within_current_boundaries': 'not established', 'confidence': 'high for observed failures only'},
        {'rank_by_observed_wall_loss': 2, 'repair_priority': 1,
         'issue': 'Local FP16 GDN selected interface requires BF16', 'attempts': 65, 'attempt_wall_seconds': 1644,
         'affected_ready_proven': 35, 'lost_card_minutes': None,
         'evidence': ['local-analysis.json', 'local-FP16_ROUTES.json', 'local-FP16_HOLDS.json'],
         'action': '35 exact temporary scheduling holds, CPU dispatch proof; no lossy dtype conversion',
         'expected_rate_gain': None, 'estimated_avoided_attempt_seconds': 35 * 1644 / 65,
         'confidence': 'high; direct route and source contract proof', 'cost_risk': 'low; reversible exact scheduling data'},
        {'rank_by_observed_wall_loss': 3,
         'issue': '16 local aclnnCat failures across heterogeneous operators', 'attempt_wall_seconds': 1450,
         'action': 'Retain exact failures; insufficient common cause proof for a broad repair or hold',
         'expected_rate_gain': None, 'confidence': 'error commonality established, shared root cause unknown'},
        {'rank_by_observed_wall_loss': 4, 'repair_priority': 2,
         'issue': 'SGL original FP16 GDN cumsum stores FP16 into a FP32 block pointer',
         'attempts_in_readonly_ranking': 33, 'attempt_wall_seconds': 1051,
         'affected_ready_proven': {'server1': holds['server1'], 'server2': holds['server2']},
         'action': 'Exact FP16 current-route temporary holds; preserve BF16 and other routes',
         'source_proof': 'chunk.py requests default cumsum FP32 output; cumsum.py stores b_o.to(s.dtype.element_ty). Both remotes identical source hashes.',
         'evidence': ['server1-analysis.json', 'SGL_CUMSUM_SOURCE.json', 'SGL_CUMSUM_HASHES.json', 'server1-FP16_ROUTES.json', 'server2-FP16_ROUTES.json'],
         'lost_card_minutes': None, 'expected_rate_gain': None,
         'estimated_avoided_attempt_seconds': (holds['server1'] + holds['server2']) * 1051 / 33,
         'confidence': 'high for shared deterministic source/type path; held unexecuted cases labelled inferred, not measured failures',
         'cost_risk': 'low; no compiler/kernel edit and no service restart'},
        {'rank_by_observed_wall_loss': 5, 'issue': 'PTO adapter pins a historical literal guard owner',
         'attempts': 11, 'attempt_wall_seconds': 376, 'initial_ready_affected': 5,
         'action': 'Diagnosed lower-priority integration restriction. No forged identity, lock bypass, source-pin relaxation or unchanged retry.',
         'evidence': 'DEFERRED_GUARD.json', 'expected_rate_gain': None, 'status': 'deferred'},
    ],
    'verification': {
        'previous_admission_hold_skips': effects['local'].get('previous_356_hold_consumption'),
        'previous_admission_hold_device_attempts': len(effects['local'].get('previous_356_device_attempts', [])),
        'new_holds_loaded_by_host': {h: effects[h]['holds_present_in_live_projection'] for h in hosts},
        'new_holds_actually_consumed': {h: effects[h]['held_consumption'] for h in hosts},
        'new_production_after_action': {h: effects[h]['production_after_action'] for h in hosts},
        'accepted_throughput_gain_attributable_to_this_action': None,
        'note': 'Loaded hold lists and unrelated cache growth are not proof of a higher accepted rate. No restarts performed.',
    },
    'remaining_at_post_snapshot': {h: {k: after[h][k] for k in ['cached', 'ready', 'held', 'tested_noncache_unheld']} for h in hosts},
    'remaining_scope_note': 'Ready excludes cached, exact temporary holds and already attempted failures. Queue-drain ETA is not all-pairs successful-completion ETA.',
    'unchanged': ['baseline kernel/compiler/schedule/checker', 'guard implementation and provenance',
                  'full outputs/input protection/3warmups/15profiler repeats/1GiB flush/no Events',
                  'PLAN ownership', 'valid caches', 'foreign processes'],
    'followup': [
        'Continue three supervisors; verify local35 new holds load at natural pair boundary and all111 holds consume without device attempts.',
        'Do not rerun failed cases without recorded real environment/route change in RETRY revision.',
        'Held FP16 routes are temporary availability restrictions; no new numerical-failure verdict for unexecuted cases.',
        'Existing BF16 frozen-source27, FFT10, admission356, server1 DeepGEMM47 and other exact holds remain.',
        'Keep server1 disk observation: approximately5.24GiB free and ~0.13GiB decrease since prior inspection; no foreign deletion.',
        'If queues finish before next check, distinguish finished testing from all pairs successfully cached.',
    ],
    'next_check': next_check,
}
assert not (D / 'RESULT.json').exists()
save(D / 'RESULT.json', record)
entry = {k: record[k] for k in ['kind', 'due_at_utc', 'checked_at', 'checked_at_utc', 'report',
    'assignment_revision', 'scope_count', 'audit_usable', 'audit_rejected', 'accepted_last_hour',
    'accepted_last_four_hours', 'new_exact_holds', 'verification', 'next_check']}
entry['audits'] = audits
with (B / 'INSPECTIONS.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (B / 'INSPECTIONS.jsonl').open('a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    save(B / 'LAST_INSPECTION.json', entry)
with (B / 'RECURRING_ALARMS.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    p = B / 'RECURRING_ALARMS.json'; a = read(p)
    match = [m for m in a['milestones'] if abs(m['due_at'] - 1793218544.2140849) < 0.01]
    assert len(match) == 1
    match[0].update(checked_at=now, inspection_report=str(D / 'RESULT.json'))
    assert any(abs(m['due_at'] - 1793232944.2140849) < .01 for m in a['milestones'])
    a['message'] = a['message'].replace('下次20:15UTC', '下次2026-10-29 00:15UTC')
    a['message'] += '\n20:15巡检已完成：inspection-20261028T201544/RESULT.json。新增精确暂挂111项：local35个ops-transformer FP16 GDN原接口只收BF16；server1 42/server2 34个SGL FP16 GDN原cumsum输出FP32却storeFP16（两端源码哈希一致，server1重复33次证据），保留BF16及其它路径。未修改原kernel/compiler/checker/guard，未重启服务。核对这些暂挂的真实清单消费和CACHE增长；local可能需当前大DSA正常结束才读新清单。旧356项已有235项跳过且0次设备重试。PTO guard身份错误为旧registration固定历史owner，不是硬件锁空闲排队；DEFERRED_GUARD.json有证据，禁止伪造owner或解除pin。所有已测失败不无变化重跑；仅推断暂挂不冒充执行失败或成功。\n'
    save(p, a)
state_path = ROOT / '.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with state_path.with_name('.state.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = read(state_path)
    m = state['baseline_cache_20261027']
    save(D / 'MEMORY_before_baseline_cache.json', m)
    m.update(updated_at=utc, status='three_host_cache_sweep_running', last_inspection=entry,
        recurring_next_due_at=next_check,
        current_ready={h: after[h]['ready'] for h in hosts},
        current_cached={h: after[h]['cached'] for h in hosts},
        current_held={h: after[h]['held'] for h in hosts},
        current_counts_observed_at={h: after[h]['host_time'] for h in hosts})
    m['active_followup'] = 'Three native CPU supervisors continue, no new agents/fleet. Next2026-10-29 00:15UTC idle-only anchor. Verify new111 exact FP16 hold consumption; completed queue is not universal success. Use latest RESULT evidence.'
    m['last_exact_fp16_hold_inspection'] = {
        'report': str(D / 'RESULT.json'), 'holds': holds,
        'source_and_dispatch_evidence': 'CPU probes prove current exact routes; original SGL cumsum/FP16 type mismatch same on both remotes; ops-transformer interface BF16-only.',
        'previous_admission_hold_effect': '235 exact skips and no device reruns observed',
        'new_hold_effect': record['verification'],
        'no_algorithm_changes': True,
        'lesson': 'Rank observed failure wall time separately from avoidable startup waste and repairability. Do not repair numerical/compiler math to inflate success; preserve source gates. A temporary hold inferred from shared source is not an executed failure verdict.',
    }
    save(state_path, state)
print(json.dumps({k: v for k, v in entry.items() if k not in ['audits', 'verification']}, ensure_ascii=False, indent=2))
