import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

P = Path(__file__).resolve().parent
selection = json.loads((P / 'SELECTION.json').read_text())
index = json.loads((P / 'EVIDENCE_INDEX.json').read_text())
best = {r['measurement_id'] for r in selection['web_best_gt_1_5']}
rows = []
for i, public in enumerate(selection['all_pair_gt_1_5']):
    mid = public['measurement_id']
    entry = index[mid]
    r = {k: public[k] for k in ['library', 'op', 'case', 'measurement_id', 'protocol', 'latency_us', 'tileops_latency_us', 'ratio']}
    r['is_web_best_gt_1_5'] = mid in best
    r['family'] = public['workload'].get('family')
    r['flags'] = []
    if entry['kind'] == 'measurement':
        path = Path(entry['directory']) / 'measurement.json'
        r['evidence'] = str(path)
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == mid
        m = json.loads(path.read_text())
        t = m.get('timing', m)
        profiles = t['retained_profiles']
        assert len(profiles) == 15
        total = collections.Counter()
        count = collections.Counter()
        cpu = collections.Counter()
        cpu_count = 0
        window_sums = []
        duplicates = 0
        for profile in profiles:
            tasks = profile['profile_kernel_tasks']
            identities = [(x['task_id'], x.get('context_id'), tuple(x.get('stream_ids', []))) for x in tasks]
            duplicates += len(identities) - len(set(identities))
            window_sums.append(sum(x['duration_us'] for x in tasks) / 1000)
            for x in tasks:
                name = x['name']
                total[name] += x['duration_us'] / 15
                count[name] += 1 / 15
                if 'AICPU' in x['core_roles']:
                    cpu[name] += x['duration_us'] / 15
                    cpu_count += 1
        r['retained_windows_checked'] = 15
        r['aicpu_count_per_call'] = cpu_count / 15
        r['aicpu_us'] = sum(cpu.values())
        r['aicpu_fraction'] = r['aicpu_us'] / public['latency_us']
        r['aicpu_tasks'] = dict(cpu)
        r['top_tasks'] = [{'name': name, 'mean_us': us, 'count_per_call': count[name]} for name, us in total.most_common(8)]
        r['kernel_count_per_call'] = sum(count.values())
        r['max_task_sum_sample_error_ms'] = max(abs(a-b) for a,b in zip(window_sums,t['latency_samples_ms'],strict=True))
        r['duplicate_task_identity_count'] = duplicates
        if r['max_task_sum_sample_error_ms'] > 1e-7 or duplicates:
            r['flags'].append('task_sum_or_identity_error')
        if cpu_count:
            r['flags'].append('aicpu_present')
        if r['aicpu_fraction'] >= 0.5:
            r['flags'].append('aicpu_dominant')
        if r['kernel_count_per_call'] >= 100:
            r['flags'].append('many_kernel_calls')
        copies = sum(us for name, us in total.items() if any(x in name.lower() for x in ['cast', 'transdata', 'transpose', 'tensormove', 'stridedslice', 'broadcastto', 'gather', 'pad']))
        r['conversion_copy_index_us'] = copies
        r['conversion_copy_index_fraction'] = copies / public['latency_us']
        if r['conversion_copy_index_fraction'] > 0.5:
            r['flags'].append('conversion_copy_index_dominant_review_semantics')
        r['profile_status'] = 'all_15_retained_task_lists_checked'
    else:
        m = entry['evidence']
        t = m['timing']
        r['evidence'] = entry['catalog']
        r['cache_key'] = m['cache_key']
        r['host'] = m['host']
        r['campaign'] = m['campaign']
        r['admission_metric'] = m['admission_metric']
        r['kernel_count_per_call'] = t['profile_kernel_count']
        r['end_to_end_mean_ms'] = m.get('end_to_end_mean_ms')
        r['profile_status'] = 'v2_aggregate_only_no_task_list_in_published_evidence'
        r['flags'].append('v2_task_coverage_unresolved')
        if t.get('profile_aic_cycles', 0) and t.get('profile_aiv_cycles', 0):
            r['flags'].append('v2_mixed_aic_aiv_sum_risk')
        if m.get('end_to_end_mean_ms'):
            r['device_sum_over_host_mean'] = t['latency_ms'] / m['end_to_end_mean_ms']
            if r['device_sum_over_host_mean'] > 1.1:
                r['flags'].append('device_sum_exceeds_separate_host_mean_10pct')
    samples = t['latency_samples_ms']
    r['samples'] = len(samples)
    r['mean_matches_web'] = math.isclose(statistics.mean(samples) * 1000, public['latency_us'], rel_tol=1e-9)
    r['coefficient_of_variation'] = statistics.stdev(samples) / statistics.mean(samples)
    r['cold_host_ms'] = t.get('cold_latency_ms')
    if not r['mean_matches_web']:
        r['flags'].append('published_mean_mismatch')
    if r['coefficient_of_variation'] > .1:
        r['flags'].append('sample_cv_over_10pct')
    rows.append(r)
    if i % 200 == 0:
        print('analyzed', i, flush=True)
    del m

(P / 'ROWS.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
summary = {
    'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'snapshot_sha256': hashlib.sha256((P / 'live.json').read_bytes()).hexdigest(),
    'rows': len(rows), 'web_best_rows': len(best),
    'flags': dict(collections.Counter(flag for r in rows for flag in r['flags'])),
    'best_flags': dict(collections.Counter(flag for r in rows if r['is_web_best_gt_1_5'] for flag in r['flags'])),
    'aicpu_ops': dict(collections.Counter(r['op'] for r in rows if 'aicpu_present' in r['flags'])),
    'many_kernel_ops': dict(collections.Counter(r['op'] for r in rows if 'many_kernel_calls' in r['flags'])),
}
(P / 'SUMMARY.initial.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
