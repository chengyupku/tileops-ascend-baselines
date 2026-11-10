import json,csv,pathlib,collections,math,datetime
W=pathlib.Path('/root/dyq/workspace');P=pathlib.Path(__file__).resolve().parent
R=W/'reports/baseline-status-20261026'
latest=W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/reconciled_scope_status0827/PRIORITY1751_ROWS.json'
inventory=json.load(open(latest));assert len(inventory)==1751
info={tuple(r[k] for k in ('baseline','operator','case_id')):r for r in csv.DictReader(open(R/'ALL_CASES.csv',encoding='utf-8-sig'))}
recent={tuple(r['key']):r for r in json.load(open(W/'reports/baseline-throughput-audit-20261026/CLASSIFICATION_METRICS.json'))['rows']}
reused=[];pending=[]
for r in inventory:
 key=tuple(r['key']);v=info[key];decision=None;source=str(latest)
 if r['joined_status']=='excluded_non_target':decision='不要接'
 elif r['joined_status']=='original_passed':decision='要接'
 rr=recent.get(key)
 if rr and rr['category']=='no':decision='不要接';source=rr['evidence']
 elif rr and 'verified' in rr['label'] and 'unverified' not in rr['label']:decision='要接';source=rr['evidence']
 if decision:reused.append({'key':list(key),'decision':decision,'basis':'retained_existing_definitive_result','evidence':[source],'new_classification':False})
 else:pending.append({'key':list(key),'family':v['family'],'source_root':str(W/'ascend_baselines'/key[0]),'provider_directory':v['shared_provider_directory_local'],'owner_status':r['owner_status'],'recent_disposition':rr,'evidence_refs':json.loads(v['all_evidence_references'])})
pending.sort(key=lambda r:(r['key'][0],r['family'],r['key'][1],r['key'][2]))
(P/'REUSED.json').write_text(json.dumps(reused,ensure_ascii=False,indent=2))
(P/'PENDING.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2))
policy='''你是本项目 baseline–kernel pair 的纯分类 worker，思考强度必须为 medium。
当前用户任务：对尚未归类的精确 (baseline库, operator, case_id) 只给“要接”或“不要接”，全队目标500个新增pair/小时。实际baseline对接已由用户暂停。
项目：TileOPs 205个算子/2062个原始case，Ascend950，各baseline供应商原版API。用户允许顶层shape/dtype/layout/调用参数适配，禁止修改baseline内部kernel、数学、schedule、compiler/pass/lowering。历史已成功3619项不再审计，本包仅含尚未二分的项。
判断边界：
1. 要接：原版完整API/示例可满足原始算子合同，允许顶层参数/形状/类型/布局适配。仅打包、缺二进制、依赖、路径、资源不足、部署失败、尚未验证，不是“不要接”的理由；记录对应障碍在reason即可。它表示保留接入候选，不宣称已经实现或数值通过。
2. 不要接：有具体源码/完整API证据表明缺失所需完整算子、数学/输出/状态/精度语义不兼容，或必须改内部实现、调度、编译器才能支持本平台/原case。不得由同族一个失败推断全族，也不得只把历史unsupported标签当证据。
3. 源码/资料不足以证明不兼容时，保守选“要接”，reason清楚写“保留候选，尚待确认X”，confidence设low。这是候选保留决策，不能伪称API已证实。没有第三种decision值。
方法：先读AGENTS/MEMORY索引及本SCOPE；不要加载长期历史。按库/算子复用一次源码事实，再逐case核shape、dtype、参数与输出。阅读原版ascend_baselines源文件、TileOPs/src/tileops/manifest/spec合同、既有provider作为线索；provider存在不能替代原版API。近期证据已附，远端路径不可用时读本地原版源，不做SSH迁移、部署、证据追溯工程。稳定同族证据可以引用同一文件与符号。
绝对禁止：NPU访问或初始化导入、执行baseline、编译、测试、benchmark、修包、改任何源码、适配器开发、等待设备锁、派遣子agent、请求审批、长篇报告、重复已完成scope。只允许CPU文本/AST/JSON读取和自己输出目录写入。不要联网查询额度。不要将旧提示词中上板验证要求带入本任务。
产物：本批每个key恰好一条，逐条追加到 decisions.jsonl，不要等全批结束才落盘。字段：key(3字符串)、decision(仅要接/不要接)、reason(简短具体)、evidence(文件路径:行号或symbol数组)、confidence(high/medium/low)、decided_at(UTC ISO)。不得以凑速率随意写无依据“不接”。可用少量静态脚本复用证据，但不重建审查框架。完成写 DONE.json {keys:N, complete:true}，final只报数量。总体目的是迅速形成接入范围，不是完成实际接入。
'''
(P/'WORKER_POLICY.txt').write_text(policy)
groups={i:[] for i in range(4)}
for idx in range(32):
 rows=pending[len(pending)*idx//32:len(pending)*(idx+1)//32]
 if not rows:continue
 group=idx//8;d=P/f'group{group+1}'/f'batch{idx+1:02d}';d.mkdir(parents=True,exist_ok=True)
 (d/'SCOPE.json').write_text(json.dumps({'rows':rows,'count':len(rows),'exclusive':True},ensure_ascii=False,indent=2))
 (d/'TASK.txt').write_text(policy+f'\n你的唯一写入目录：{d}\n你的唯一输入scope：{d}/SCOPE.json\n从现在直接做分类，禁止实际对接。\n')
 groups[group].append({'batch':idx+1,'directory':str(d),'count':len(rows),'task_file':str(d/'TASK.txt')})
for i,items in groups.items():
 d=P/f'group{i+1}';(d/'BATCHES.json').write_text(json.dumps(items,indent=2))
 (d/'COORDINATOR_TASK.txt').write_text(f'''Use codex-workers skill. You are coordinator {i+1} for CPU-only baseline classification. User explicitly authorizes 32 classifier agents, 4 real parent groups of8. You may delegate ONLY the8 disjoint batches in {d}/BATCHES.json using the existing native codex-worker lifecycle. ALL coordinators and workers must be gpt-6-astra with effort medium. No NPU, no integration/build/tests/source edits, no quota RPC, no memory writes, no other task writes.
Immediately run: python3 {P}/dispatch_group.py {d}
This uses the existing approved manual-quota selection and native supervisor. Do not reinvent dispatch or query quotas. It writes GROUP_JOBS.json, each child has its exact separate batch write directory. No ninth concurrent child. Do not change worker prompts; root prepared the binary decision boundary/context.
After dispatch, wait and consume original children using native status/inbox or read GROUP_JOBS/native DB/events every30-60s. Do not busy-poll or audit every decision. Preserve outputs and report mechanical errors; do not retry or switch accounts on provider/model error. The main agent diagnoses shared failures. Stay available until all children terminal or a user stop. You may inspect only your group output and read shared policy/source. When all children terminate, write GROUP_DONE.json with statuses/counts, collect inbox notifications, then finish concise. Do not launch further rounds or do classification yourself; immediate useful parallel launch is your task.
''')
plan={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope_total':1751,'reused':len(reused),'new_scope':len(pending),'groups':4,'max_classifiers':32,'effort':'medium','model':'gpt-6-astra','target_new_pairs_per_hour':500,'monitor_seconds':3600,'decision_values':['要接','不要接'],'actual_integration_paused':True,'authority':'Explicit current user instruction; no NPU for classification; retained source boundary.'}
(P/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));print(json.dumps(plan,ensure_ascii=False))
