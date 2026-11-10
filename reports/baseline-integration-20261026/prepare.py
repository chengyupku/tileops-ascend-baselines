import pathlib,json,csv,datetime
P=pathlib.Path('/root/dyq/workspace');D=P/'reports/baseline-integration-20261026';C=P/'reports/baseline-classification-20261026'
verified={};
for f in D.glob('*_verified.json'):
 for x in json.load(open(f))['receipts']:verified.setdefault(tuple(x['key']),[]).append(dict(x,host=f.stem.replace('_verified','')))
rows=list(csv.DictReader(open(C/'NEW_DECISIONS.csv',encoding='utf-8-sig'))); pending={tuple(x['key']):x for x in json.load(open(C/'PENDING.json'))}
groups=[('nn_norm','local','ops-nn',['RMSNormFwdOp','GroupNormFwdOp','InstanceNormFwdOp','LayerNormFwdOp']),('nn_matrix','local','ops-nn',['GemmFwdOp','GemmINT8W8A8FwdOp']),('nn_misc_math','local',None,['BatchNormBwdOp','CountNonzeroFwdOp','SoftmaxFwdOp','VectorNormFwdOp','SumFwdOp']),('transformer_dense','server1','ops-transformer',['GemmFwdOp','BmmFwdOp']),('transformer_grouped','server1','ops-transformer',['GroupedGemmFwdOp','MoEGroupedGemmFwdOp']),('transformer_gdn','server2','ops-transformer',['GDNFwdOp','DeltaNetFwdOp']),('transformer_attention','server2','ops-transformer',['GQADenseFwdOp']),('tla_direct','server1','tilelang-ascend',['CausalConv1dUpdateFwdOp','RMSNormFwdOp','WhereFwdOp'])]
policy='''用户最新授权：停止吞吐量监控，派出“一定要接但还没接”的实际 baseline 对接任务，多 agent 并行。此前暂停被这份有限清单取代；旧 HOLD、旧调度器保持不动。不是重新做分类，也不是性能优化。
项目：TileOPs Ascend950 baseline 与 canonical kernel case 对接。每个 key=(baseline,operator,case_id)；SCOPE.json 是排他分工，不扩大范围。候选已有完整原版 API 的静态高置信证据；它不保证构建/数值已通过。真实源代码优先于分类 reason。如发现分类错误或必须改原算法，保留 blocked/needs_review 并说明，禁止强行接。
边界：只允许顶层参数、合法 dtype/layout/shape/元数据适配、调用原版完整 API、挑选合同输出。禁止改 vendor kernel、数学公式、内部 tile/block/schedule、编译器；禁止把多算子数学分解/新的算法伪装成顶层适配。禁止 PyTorch oracle/fallback 冒充 baseline。不得降低精度/缩小输入/改 seed、容差或删除输出校验。
验收：可调用的真实原版入口+本 scope canonical case 完整正确性。复用现有环境、原版库、现成 launch/run_case.py 与 ascend_pair_check.py；先读对应历史最近 blocker 和 source evidence，复用成功路线。不是写一份建议就结束；能接的完成适配、运行、留证。完整 output/final_state/cache 更新都检查。无需 benchmark/profiling/autotune。
去重：根已从三台当前结果中剔除报告 verified 的 exact keys。执行前再只查自己 key 的最新 receipts、目标集和用户 amnesty；已完成的不重跑，不做旧成功重新认证。取消时留下的成功 case 优先保留；失败路线若无输入/环境修复证据不原样重试。共性 blocker 出现后及时停同路线，写具体错误与证据，不以重建整套编译器消耗时间。
隔离：只能写本任务目录及指定远端同名私有目录。共享 TileOPs/provider/vendor/环境/记忆均只读。把需要的 provider 顶层文件复制到私有目录，使用现有 overlay/runner 机制；或本任务目录建立独立 TileOPs 副本（保留未改 verifier），在私有副本完成真正可调用注册与验证。交付 adapter、精确补丁、可复现 guarded launch 和 DELIVERY.json，供主 agent 合并。不要并发写共享 provider/registry；不要提交或覆盖他人工作。
NPU 明确获准，仅限指定 host。唯一 owner 见下。任何初始化、torch_npu 导入、device tensor、编译中隐含初始化、正确性运行必须由 npu-guard run --owner OWNER --wait 3600 --timeout 300 -- COMMAND 包住整个前台命令；不准后台逃逸/嵌套预约/裸跑。CPU 准备不可占锁。读 docs/NPU共享锁.md，75/76/78 不得绕过。禁止重置设备或结束其他任务进程。每个 agent 最多一个待执行 device 请求，短批次 amortize 导入，每次最长300秒。
每个 native agent effort=medium，禁止派额外 child、禁止改模型/额度查询。原生监督器自动向主会话 queue 投递完成通知，无需自写通知或监控器。不要启动任何长期监控。继续处理 scope 剩余可行项直到完成或明确阻塞；不要对共性失败逐 case 重复浪费。
输出：先在本目录写 STARTED.json（host、scope count、计划原版API），最后 RESULT.json（每key status、api、changed_files、guard/checker路径、checked/total outputs、已完成跳过/阻塞原因）与 DELIVERY.json（可调用入口、运行方法、patch、未解决项）。通过必须有真实完整checker与guard退出0；不得把静态检查或CLI completed当数值成功。远端结果复制回本地任务目录，原路径也保留。
'''
(D/'POLICY.txt').write_text(policy)
manifest=[]; skipped=[]
for name,host,lib,ops in groups:
 d=D/name;d.mkdir(exist_ok=True); selected=[]
 for r in rows:
  k=(r['baseline'],r['operator'],r['case_id'])
  if r['confidence']!='high' or r['decision']!='要接' or r['operator'] not in ops or (lib and r['baseline']!=lib):continue
  if lib is None and r['baseline'] not in ['ops-nn','ops-math']:continue
  if k in verified:skipped.append({'key':k,'receipts':verified[k]});continue
  selected.append(dict(r,key=list(k),context=pending[k]))
 (d/'SCOPE.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
 remote='/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/'+name
 hostalias={'local':'local','server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host]
 owner='selected-integration-20261026/'+name
 hints='本地最近正常 norm 启动样板：.cache/all-pairs/full_rollout_20261024/local24/group1/resume25_common_repair/continuation0320/tasks/current-original-nn-norm-launch-repair20/packet/launch.py；同目录 current-original-tla-norm-validate25 已25通过，不重跑。远端 server2 最近恢复路径：.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks/recovery0730-*；gdn-original-e2e-recovered / recovery0730-gdn-fixed-abi / recovery0730-gdn-q35-small-abi。server1 现成任务在 .cache/all-pairs/full_rollout_20261024/remote_ready/server1/group*/recovery_resume25。只按本key查证，勿遍历全部历史。'
 task=policy+f'\n本任务 name={name}; host={host}; owner={owner}; scope={d}/SCOPE.json; 本地可写={d}。\n'+hints+'\n'
 if host!='local':task+=f'远端通过已有SSH wrapper /root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh {hostalias} 访问。远端项目根 /opt/tileops-all-pairs/workspace，可写仅 {remote}；需要目录请mkdir。不要另派远端agent；你直接在远端执行自己的任务，队列通知仍由本地native负责。SCOPE中 /root/dyq/workspace 前缀在远端换成 /opt/tileops-all-pairs/workspace；证据路径按原host读取，不假装跨主机文件存在。\n'
 else:task+='本地原版环境入口 .tools/baselines/baseline-run LIBRARY python ...。\n'
 (d/'TASK.txt').write_text(task)
 manifest.append({'name':name,'host':host,'owner':owner,'directory':str(d),'task_file':str(d/'TASK.txt'),'count':len(selected),'remote_write':remote if host!='local' else None})
(D/'SKIPPED_ALREADY_VERIFIED.json').write_text(json.dumps(skipped,ensure_ascii=False,indent=2))
(D/'PLAN.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'用户最新：停止监控，派出一定要接但还没接的任务，多agent；原队列仍暂停，仅此清单恢复。','model':'gpt-6-astra','effort':'medium','workers':manifest,'selected_count':sum(x['count'] for x in manifest),'skipped_verified_count':len(skipped),'monitoring':False},ensure_ascii=False,indent=2))
print(json.dumps({'workers':manifest,'skipped':len(skipped)},ensure_ascii=False,indent=2))
