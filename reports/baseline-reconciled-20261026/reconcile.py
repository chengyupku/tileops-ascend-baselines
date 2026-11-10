import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'reports/baseline-status-20261026'
CLASS = ROOT / 'reports/baseline-classification-20261026'
NEW = ROOT / 'reports/baseline-integration-20261026'
LEDGER = ROOT / '.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/reconciled_scope_status0827/EXACT_TARGET_SETS.jsonl'

def readcsv(p):
    return list(csv.DictReader(p.open(encoding='utf-8-sig')))

def key(row):
    return tuple(row['key']) if 'key' in row else tuple(row[x] for x in ('baseline', 'operator', 'case_id'))

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def writecsv(name, rows, fields=None):
    rows = list(rows)
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

old = {key(x): x for x in readcsv(OLD / 'ALL_CASES.csv')}
legacy = {key(x) for x in readcsv(OLD / 'provisionally_retained_legacy_success.csv')}
ledger = {key(x): x for x in map(json.loads, LEDGER.open())}
done = {k for k, x in ledger.items() if x['classification'] == 'original_supported_checked'}
initial_done = done.copy()
decisions = {key(x): x for x in readcsv(CLASS / 'DECISIONS_WITH_EVIDENCE.csv')}
selected = {key(x): x for x in readcsv(NEW / 'FINAL_CASES.csv')}
old_pending = {key(x) for x in readcsv(OLD / 'retained_not_passed.csv')}
guards = {key(x): x for h in ('local', 'server1', 'server2') for x in json.load((OUT / f'{h}_guards.json').open())}
fresh = set()
incomplete = set()
for k, x in guards.items():
    s = x['summary']
    v = s['verification']
    assert s['status'] == 'verified' and v['checked_outputs'] == v['total_outputs'] > 0
    assert s['inputs_preserved'] and not s['provider_source_changed'] and not s['oracle_source_changed']
    if any(g.get('rc') == 0 for g in x['guards']):
        fresh.add(k)
    else:
        incomplete.add(k)
done |= fresh
done |= {k for k, x in selected.items() if x['status'] in ('registered_verified', 'retained_prior')}
blocked = {k for k, x in selected.items() if x['status'] == 'blocked'}
pending = (old_pending | blocked | incomplete) - done - legacy
excluded = {k for k, x in old.items() if x['category'] in ('excluded_old_success', 'excluded_no_old_success')}
excluded |= {k for k, x in decisions.items() if x['decision'] == '不要接'}
candidates = {k for k, x in decisions.items() if x['decision'] == '要接'} - done - legacy - pending
sets = [done, legacy, pending, candidates, excluded]
assert len(old) == 10833
assert set.union(*sets) == set(old)
assert sum(map(len, sets)) == len(old), 'overlapping classifications'
assert len(done) == 2459 and len(legacy) == 3619
assert len(pending) == 295 and len(candidates) == 611 and len(excluded) == 3849

rows = []
for k in sorted(old):
    source = ''
    reason = ''
    if k in legacy:
        state, basis = 'completed_retained', '用户明确暂留旧成功；本次未重审'
        source = str(OLD / 'USER_AMNESTY.json')
    elif k in done:
        state, basis = 'completed_original_verified', '原版合规成功账本'
        source = str(LEDGER)
        if k in fresh:
            basis = '账本外新增原版全输出成功；guard退出0'
            source = guards[k]['checker']
        if k in selected:
            basis = '本批新增56正式接入' if selected[k]['status'] == 'registered_verified' else '本批复核保留既往成功1项'
            source = selected[k]['registration_review'] or selected[k]['result']
    elif k in pending:
        state = 'definite_pending'
        if k in old_pending:
            basis, reason = '历史已确认目标219项', '仍无当前合规成功；历史旧回执不能覆盖后续资格判定'
            source = old[k]['qualified_evidence_references']
        elif k in blocked:
            basis, reason = '本批已确认目标但受阻75项', '见对应任务RESULT与FINAL_DELIVERY阻塞分类'
            source = selected[k]['result']
        else:
            basis, reason = '数值成功但验收证据不全1项', '缺少对应guard最终退出回执；暂不计完成，不自动重跑'
            source = guards[k]['checker']
    elif k in candidates:
        state, basis = 'candidate_to_confirm', '二元分类要接；尚未证明完整原版接口可用'
        reason, source = decisions[k]['reason'], decisions[k]['evidence']
    else:
        state, basis = 'excluded', '无需对接'
        reason = decisions[k]['reason'] if k in decisions else old[k]['reason_tag']
        source = str(CLASS / 'DECISIONS_WITH_EVIDENCE.csv') if k in decisions else str(OLD / 'ALL_CASES.csv')
    rows.append(dict(zip(('baseline', 'operator', 'case_id'), k), status=state, basis=basis, reason=reason, evidence=source))
writecsv('ALL_PAIRS.csv', rows)
writecsv('COMPLETED.csv', (x for x in rows if x['status'].startswith('completed')))
writecsv('DEFINITE_PENDING.csv', (x for x in rows if x['status'] == 'definite_pending'))
writecsv('CANDIDATES_TO_CONFIRM.csv', (x for x in rows if x['status'] == 'candidate_to_confirm'))
writecsv('EXCLUDED.csv', (x for x in rows if x['status'] == 'excluded'))
by = defaultdict(Counter)
for x in rows:
    by[x['baseline']][x['status']] += 1
fields = ['baseline', 'completed_original_verified', 'completed_retained', 'definite_pending', 'candidate_to_confirm', 'excluded']
writecsv('BY_BASELINE.csv', ({f: b if f == 'baseline' else c[f] for f in fields} for b, c in sorted(by.items())), fields)
groups = Counter((k[0], k[1]) for k in pending)
writecsv('PENDING_BY_OPERATOR.csv', ({'baseline': b, 'operator': o, 'pairs': n} for (b, o), n in sorted(groups.items())))
delivery = json.load((NEW / 'FINAL_DELIVERY.json').open())
jobs = json.load((NEW / 'JOBS.json').open())['jobs']
start = min(x['submitted_at'] for x in jobs)
end = datetime.fromisoformat(delivery['at']).timestamp()
hours = (end - start) / 3600
rate = 56 / hours
summary = {
    'at': datetime.now(timezone.utc).isoformat(),
    'key': ['baseline', 'operator', 'case_id'],
    'scope': '固定账本10833个已观察exact pair；不是baseline库数量，也不是完整笛卡尔积',
    'observed_pairs': len(old),
    'completed_including_user_retained': len(done | legacy),
    'original_verified': len(done),
    'user_retained_without_reaudit': len(legacy),
    'definite_pending': len(pending),
    'definite_pending_components': {'historical_confirmed': len(old_pending), 'selected_blocked': len(blocked), 'missing_guard_receipt': len(incomplete)},
    'candidate_to_confirm': len(candidates),
    'excluded': len(excluded),
    'original_verified_components': {'joined_ledger': len(initial_done), 'additional_full_checked_guard_zero': len(fresh), 'selected_new_registered': 56, 'selected_prior_success_reviewed': 1},
    'selected_batch': delivery,
    'pending_by_baseline': dict(Counter(k[0] for k in pending)),
    'candidate_confidence': dict(Counter(decisions[k]['confidence'] for k in candidates)),
    'eta': {'observed_new_registered': 56, 'observed_elapsed_hours': hours, 'observed_new_registered_per_hour': rate, 'definite_pending_linear_hours': len(pending) / rate, 'all_candidates_eligible_linear_hours': (len(pending) + len(candidates)) / rate, 'condition': '仅按上一批实际交付速率线性折算；需要继续运行并解决共性阻塞，非截止时间承诺', 'reliable_full_completion_deadline': None},
    'limitations': ['3619项按用户授权暂留，不声称本轮重新验证。', '全局计数按历史验收账本与补充数值证据；只有本批56项此次逐一核验正式入口，不声称6078项统一便携安装包已重新审计。', 'CPU参考验证器存在已记录版本差异；FLA保留canonical_verifier哈希及CHECKS，normalization为独立CPU参考。', '未将过期原始success回执重新提升为完成，避免覆盖较新的资格判定。', '1项checker通过但guard最终退出回执缺失，保守保留待验收。', '此整理未运行NPU、未恢复监控或旧队列。'],
    'sources': [str(LEDGER), str(OLD / 'ALL_CASES.csv'), str(OLD / 'USER_AMNESTY.json'), str(CLASS / 'DECISIONS_WITH_EVIDENCE.csv'), str(NEW / 'FINAL_CASES.csv')]
}
dump('SUMMARY.json', summary)
dump('RECEIPT_ACCEPTANCE.json', {'accepted': [guards[k] for k in sorted(fresh)], 'pending_guard': [guards[k] for k in sorted(incomplete)]})
print(json.dumps({k: v for k, v in summary.items() if k not in ('selected_batch', 'sources', 'limitations')}, ensure_ascii=False, indent=2))
