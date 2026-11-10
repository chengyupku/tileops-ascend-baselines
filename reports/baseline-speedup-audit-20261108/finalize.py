import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
rows = json.loads((P / 'ROWS.json').read_text())
replayed = json.loads((P / 'AICPU_RAW_REPLAY.json').read_text())
receipts = json.loads((P / 'RECEIPT_CHECKS.json').read_text())
assert len(replayed) == 29 and sum(len(x['windows']) for x in replayed) == 435
assert len(rows) == 2790 and all(r['mean_matches_web'] for r in rows)
assert len(receipts) == 866 and all(r['guard_valid'] and r['correctness_verified'] for r in receipts)

def write_csv(name, data, columns):
    with (P / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in data:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items() if k in columns})

groups = [
    ('P1-int4', 'P1', 'INT4 打包', {'INT4QuantPerGroupFwdOp'},
     'LeftShift / BitwiseAnd AICPU 占99.61%–99.66%；本轮前已在本地验证算术打包可消除AICPU。',
     '保留量化与舍入，改用16*high+where(low<0,low+16,low)，最后转int8。',
     '补物理32核8case正式复测；保留全部256种编码、常量组、scale/offset与原checker。',
     'reports/torch-all2208-20261105/source/workloads/quantization/quantize.py:75'),
    ('P1-complex-logical', 'P1', '复数逻辑归约', {'AllFwdOp', 'AnyFwdOp', 'CountNonzeroFwdOp'},
     'complex64/complex128转bool或NotEqual落AICPU，占99.46%–99.71%。',
     '评估实部/虚部分别判断非零再OR，随后原逻辑归约；也可研究位级非零判定。',
     '必须保留NaN/Inf、正负零及极小非零；complex128拆成float64后仍可能AICPU，禁止随意降为float32。未设备验证替代式。',
     'reports/torch-all2208-20261105/source/workloads/reduction.py:470'),
    ('P1-mask-pack', 'P1', 'tilelang-ascend mask打包', {'MaskedFillScalarFwdOp', 'MaskedFillTensorFwdOp', 'WhereFwdOp'},
     '适配层8次uint8 LeftShift占96.48%–96.64%，原生VSEL不是主要瓶颈。',
     '评估布尔位乘1/2/4/.../128再求和并转uint8，替代移位打包，保留原VSEL。',
     '所有布局转换继续计时；校验bit顺序、广播、尾部和非规范bool存储真值。未验证候选速度。',
     'reports/baseline-flex-20261106/server1/tla_adapters_v2.py:142'),
    ('P1-ones', 'P1', 'tilelang-ascend LogicalNot常量生成', {'LogicalNotFwdOp'},
     'OnesLike AICPU约2.044ms，占58.14%；其余原kernel与转换仍需计入。',
     '核对int16 ones分配/填充值路径，评估等价fill或全速dtype创建后cast。',
     '保留原bitwise_xor内核，确认实际profile不再AICPU；常量创建不能移出计时伪造收益。',
     'TileOPs/benchmarks/ascend_all_pairs/tla/adapters.py:104'),
    ('P2-cumprod', 'P2', 'Cumprod原生慢路径', {'CumprodFwdOp'},
     'torch.cumprod自身运行在AICPU，占99.95%–100%，不是单个打包表达式问题。',
     '评估同合同的设备原生cumprod或合法已有实现；先看已有更快baseline。',
     '不可直接以exp(cumsum(log(x)))替换：零、负数、溢出和舍入行为不同；保持FP32中间计算与原checker。',
     'reports/torch-all2208-20261105/source/workloads/reduction.py:266'),
    ('P2-scan-precision', 'P2', 'SelectiveScan精度适配', {'SelectiveScanFwdOp'},
     'mamba-130m-no-gate-float32启用FP64 softplus与PowAiCpu，AICPU占57.49%，同时逐步递推产生约24728个设备任务。',
     '单独研究满足既有精度的设备路径；保持独立CPU oracle，不能直接撤销精度修复。',
     '需保留softplus阈值、decay舍入、状态递推和全部输出；不得将简单降精度算作优化成功。',
     'reports/torch-all2208-20261105/adapters/scan_precise_v16.py:26'),
]
queue = []
for gid, priority, title, ops, finding, proposal, constraints, source in groups:
    selected = [r for r in rows if r['op'] in ops and 'aicpu_present' in r['flags']]
    item = dict(id=gid, priority=priority, title=title, status='confirmed_aicpu_bottleneck',
                pair_count=len(selected), web_best_count=sum(r['is_web_best_gt_1_5'] for r in selected),
                ratio_min=min(r['ratio'] for r in selected), ratio_max=max(r['ratio'] for r in selected),
                aicpu_percent_min=min(r['aicpu_fraction'] for r in selected)*100,
                aicpu_percent_max=max(r['aicpu_fraction'] for r in selected)*100,
                finding=finding, proposal=proposal, constraints=constraints, source=source,
                keys=[[r['library'],r['op'],r['case']] for r in selected])
    queue.append(item)

review_groups = [
    ('P0-v2-mixed', 'P0', '旧v2混合AIC/AIV计时复核', 'v2_mixed_aic_aiv_sum_risk',
     '原解析器重复累计同任务AIC/AIV的机制已有独立证据；本清单392条同时有AIC/AIV活动，仅为风险候选，不能逐条认定重复。',
     '先复用已验收同合同v3记录，否则按distinct-task-span重新测量；不可机械除以2。'),
    ('P2-v2-coverage', 'P2', '旧v2完整任务覆盖复核', 'v2_task_coverage_unresolved',
     '1924条发布证据仅保留聚合数据，不能证明无AICPU或完整任务覆盖；包含上述392条，不能相加。',
     '逐项绑定同合同最新完整v3证据；无证据的条目进入复测队列，不能标记已排除异常。'),
    ('P2-many-kernels', 'P2', '参考实现多调用', 'many_kernel_calls',
     '13条每次调用至少100个设备任务，SelectiveScan约8238–45201个；可能为逐步参考或组合算法成本，不等同实现错误。',
     '检查实际循环与融合边界，区分oracle与性能参考，只有同语义实现才可比较。'),
    ('P3-conversions', 'P3', '转换/复制/索引占比复核', 'conversion_copy_index_dominant_review_semantics',
     '106条按任务名称统计的cast/copy/index等占比超过50%，其中4条已计入AICPU类；必要转换不自动视作缺陷。',
     '逐项核对dtype和布局合同，可省转换才做候选；不得改变中间精度或从计时中移除必要转换。'),
    ('P3-variance', 'P3', '样本波动复核', 'sample_cv_over_10pct',
     '23条15次样本变异系数超过10%；不是已证实测量错误。',
     '保留全部样本，按设备共享协议受控复测，不挑最小值。'),
]
for gid, priority, title, flag, finding, proposal in review_groups:
    selected = [r for r in rows if flag in r['flags']]
    queue.append(dict(id=gid, priority=priority, title=title, status='review_required_not_confirmed_defect',
                     pair_count=len(selected),web_best_count=sum(r['is_web_best_gt_1_5'] for r in selected),
                     finding=finding,proposal=proposal,keys=[[r['library'],r['op'],r['case']] for r in selected]))
(P / 'FIX_QUEUE.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n')
columns=['library','op','case','is_web_best_gt_1_5','ratio','latency_us','tileops_latency_us','protocol','profile_status',
         'flags','aicpu_fraction','aicpu_tasks','kernel_count_per_call','conversion_copy_index_fraction',
         'coefficient_of_variation','device_sum_over_host_mean','mean_matches_web','measurement_id','evidence']
write_csv('ALL_PAIRS.csv',rows,columns)
write_csv('WEB_BEST.csv',[r for r in rows if r['is_web_best_gt_1_5']],columns)
write_csv('AICPU_CONFIRMED.csv',[r for r in rows if 'aicpu_present' in r['flags']],columns)
write_csv('FIX_QUEUE.csv',queue,['id','priority','title','status','pair_count','web_best_count','ratio_min','ratio_max','aicpu_percent_min','aicpu_percent_max','finding','proposal','constraints','source'])
summary=json.loads((P/'SUMMARY.initial.json').read_text())
summary.update(snapshot_url='https://yyttt6.github.io/TileOPs.github.io/benchmarks/baseline-cache.json',
               selection='Every published individual baseline/TileOPs ratio strictly >1.5; separate fastest-baseline webpage subset',
               unique_ops=130,unique_cases=994,web_best_ops=74,measurement_ids_mapped=2790,
               v3_measurements_checked=866,v3_retained_task_windows_checked=12990,
               v3_aicpu_measurements=29,v3_aicpu_ops=10,previous_int4_pairs=8,new_aicpu_pairs=21,
               aicpu_raw_windows_replayed=435,raw_replay_scope='All 29 AICPU measurements, all15 windows each',
               new_device_runs=0,production_changes=False,publication_changes=False,
               interpretation='AICPU cost is confirmed, replacement efficacy is untested except prior local INT4 probe. Aggregate-only v2 entries remain unresolved; no claim that other baselines are defect-free.',
               legacy_policy='Saved v2 values were explicitly restored on2026-11-01; this audit records risks without changing that publication policy.')
newer_path=P/'UNPUBLISHED_REMEASUREMENT_CANDIDATES.json'
if newer_path.exists():
    newer=json.loads(newer_path.read_text());summary['accepted_v3_candidates_for_v2']=len(newer)
    summary['accepted_v3_candidates_same_workload']=sum(r['same_workload'] for r in newer)
(P/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

lines=['Baseline >1.5x 审计（2026-11-08网页快照）',
       '范围：2790 baseline/operator/case条目，130算子、994 workload；网页最快baseline口径为328 workload、74算子。',
       '证据：全部2790条绑定发布身份；866条v3核验12990个保留任务窗口及正确性/guard；29条AICPU异常额外重放435个原始窗口。',
       '本轮仅CPU取证，无设备复测、正式baseline修改或网页发布。', '', '确认的AICPU耗时问题：']
for q in queue[:6]:
    lines += [f"{q['id']} {q['title']}: {q['pair_count']}条，当前网页最快对照{q['web_best_count']}条，倍率{q['ratio_min']:.2f}–{q['ratio_max']:.2f}。",
              '  '+q['finding'],'  候选：'+q['proposal'],'  约束：'+q['constraints']]
lines += ['', '计时与其他候选（类别有交集，不可相加）：']
for q in queue[6:]:
    lines += [f"{q['id']} {q['title']}: {q['pair_count']}条，网页最快对照{q['web_best_count']}条。",'  '+q['finding'],'  下一步：'+q['proposal']]
lines += ['', '10条旧v2记录的设备和超过另一组同步host均值10%，全部属于混合AIC/AIV风险；独立窗口不同，作为复核信号而非逐条证明。',
          '所有2790条发布均值与其15样本一致；866条v3未发现任务重复身份或任务耗时求和不一致。',
          '未发现AICPU不等于已证明实现最优。旧v2仅有聚合记录的1924条不能排除AICPU或覆盖缺失。',
          '优先次序：P0计时口径风险与P1打包/复数非零判断并列准备；然后处理需精度/算法验证的P2；P3仅列候选。',
          '表格：ALL_PAIRS.csv全部2790条，WEB_BEST.csv网页328条，AICPU_CONFIRMED.csv确认29条，FIX_QUEUE.csv/JSON批量处理清单。']
if newer_path.exists():
    lines += [f"已有{summary['accepted_v3_candidates_same_workload']}条旧v2条目找到同workload的新v3已验收候选，见UNPUBLISHED_REMEASUREMENT_CANDIDATES.json；先审阅可复用证据，不重复占卡。这不是自动发布许可。"]
(P/'REPORT.txt').write_text('\n'.join(lines)+'\n')
readings=['.agent_memory/PROJECTS/tileops-ascend950/LESSONS/measurement_and_environment.md',
          '.agent_memory/PROJECTS/tileops-ascend950/LESSONS/development_practice.md',
          '.agent_memory/PROJECTS/tileops-ascend950/LESSONS/execution_and_delivery.md',
          '.agent_memory/PROJECTS/tileops-ascend950/LESSONS/validation_and_reuse.md',
          '.agent_memory/PROJECTS/tileops-ascend950/LESSONS/pr_baseline_coverage.md','docs/开发经验.md']
(P/'STATE.json').write_text(json.dumps({'status':'audit_complete_with_explicit_legacy_evidence_gaps',
    'authorization':'Audit current >1.5x baselines; user plans later batch fixes',
    'readings':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in readings},
    'applicable':'measurement provenance, raw task coverage, source/contract comparison; historical campaign roles/delegation/remeasurement are not resumed',
    'next':'Use FIX_QUEUE for later authorized batch fixes and legacy v3 evidence review',
    'at':datetime.datetime.now(datetime.timezone.utc).isoformat()},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['rows','web_best_rows','v3_aicpu_measurements','aicpu_raw_windows_replayed','accepted_v3_candidates_same_workload'] if k in summary},indent=2))
