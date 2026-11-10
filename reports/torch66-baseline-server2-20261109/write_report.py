import pathlib,json,collections
P=pathlib.Path(__file__).resolve().parent;s=json.loads((P/'FINAL_RESULTS.json').read_text());rows=s['selected'];groups=collections.defaultdict(list)
for r in rows:groups[r['op']].append(r)
lines=['66 个 torch baseline 修订与 v3 复测结果','',f"完成 {s['cases']}/{s['cases']} case，覆盖 {s['ops']} 个算子。每个 case 至少满足用户四项条件之一。",f"{s['within_2x']} 个新比例 <=2x；{s['still_over_10x']} 个仍 >10x。保留高比例，不通过改变精度、输入或计时方法降低数值。",'','验收证据：每 case 原始正确性检查、有效 npu-guard 回执、15 个独立 v3 窗口、本地原始二进制重放、源码与归档 SHA256。','比较口径：新 baseline / 网页冻结的 TileOPs 设备耗时；本轮只重测 baseline。v3 设备任务耗时不包括 Python 发射间隙。','', '按算子汇总：','算子 | case 数 | 原比例范围 | 新比例范围 | baseline 改善倍数范围']
for op,rs in groups.items():
 def span(k):return f"{min(r[k] for r in rs):.2f}–{max(r[k] for r in rs):.2f}"
 lines.append(f"{op} | {len(rs)} | {span('old_ratio')}x | {span('new_ratio_with_published_tileops')}x | {span('baseline_improvement')}x")
lines+=['','仍超过 10x 的 case（已按其他条件验收）：']
for r in rows:
 if r['new_ratio_with_published_tileops']>10:lines.append(f"{r['ordinal']:02d} {r['op']} / {r['case']}: {r['old_ratio']:.2f}x -> {r['new_ratio_with_published_tileops']:.2f}x；{r['old_baseline_us']:.3f} -> {r['new_baseline_us']:.3f} us；条件 {r['accepted_goals']}。{r['review_note']}")
lines+=['','AICPU：']
if not s['remaining_aicpu_cases']:lines.append('66 个选定实现的稳态计时窗口均未发现 AICPU。')
else:
 for r in rows:
  if r['aicpu_us']>0:lines.append(f"{r['ordinal']:02d} {r['op']} / {r['case']}：平均 AICPU {r['aicpu_us']:.3f} us；为保持既有精度契约保留的 FP64 路径，本轮未消除，按消除逐步冗余与保持计算语义验收。")
lines+=['','未采纳：MoE 的 searchsorted 方案触发 AICPU；ragged MeanPooling 首版额外产生较大内存搬运。使用已验证的原生 routing、修订后的掩码/归约实现；失败和退化尝试均保留。','Cumsum：确认耗时集中在公开原生 torch.cumsum 的设备任务，改写未显著改善；仅按忠实反映框架性能验收。','','明细：RESULTS.csv / FINAL_RESULTS.json。实现入口：baseline_dispatch.py / ROUTES.json。复现与边界：DELIVERY.txt。原始证据：remote_evidence/evidence/INDEX.json。']
(P/'FINAL_REPORT.txt').write_text('\n'.join(lines)+'\n')
