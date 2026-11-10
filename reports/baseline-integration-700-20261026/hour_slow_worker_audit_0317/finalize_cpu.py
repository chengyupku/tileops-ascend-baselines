import pathlib,json,datetime,hashlib,collections
P=pathlib.Path(__file__).parent;B=P.parent;A=json.loads((P/'AUDIT.json').read_text());W={w['task']:w for w in A['workers']}
def ev(task,needle):
 return [dict(guard=r['guard'],start=r['start'],end=r['end'],window_overlap_seconds=r['window_overlap_seconds'],checker=r['checker'],checker_error=r['checker_error'],log_samples=r['log_samples']) for r in W[task]['attempts'] if needle in r['guard']]
A['findings']=[
 {'priority':1,'finding':'最大共性是部署/观测前置检查不足，以及准入等待：没有形成一次准备、一次完整验证的稳定路径。不能把全部非零退出都算环境失败。','evidence':{'nn_window_environment_failures':[r for r in W['threehost_nn11']['attempts'] if r['exit']==1 and r['window_overlap_seconds']],'pto_failures':[r for r in W['threehost_pto_original23']['attempts'] if r['exit']==1],'linear_window_failures_note':'4 exit1中3为数值不匹配、1为NoneType.shape观测错误，不能都归环境；见attempts的checker_error。','attention_environment_failures_note':'4次exit1发生于窗口之前，约688.7秒；本窗口不能再扣。','admission_75_window_work_seconds':sum(w['window_attempt_summary']['guard_work_seconds_by_exit'].get('75',0) for w in A['workers']),'admission_75_window_attempt_count':sum(w['window_attempt_summary']['exit_counts'].get('75',0) for w in A['workers'])},'limit':'工作秒可跨host/worker并行，非本小时墙钟损失，不能作为利用率或吞吐差额的直接因果分解。'},
 {'priority':2,'finding':'输出与历史读取过宽，四任务623完成命令、4583190字符；PTO全部在窗口内，单条整份SDK assembly收据输出1048606字符。','evidence':[{'events':w['events']['path'],'worker_lifetime':w['lifetime'],'largest_outputs':w['events']['largest_outputs'][:2]} for w in A['workers']],'limit':'其他3任务窗口内命令/字符未知；输出量不能换算模型耗时。命令完成不算数值成功。'},
 {'priority':3,'finding':'attention额外全局stage profiler/线程设置相关试验取消，未产出完整checker；窗口内覆盖638.78秒。','evidence':ev('threehost_attention_moe53','attempt05')+[{'path':'/root/dyq/codex-workers/jobs/f1ecbd644c6d4324/events.jsonl','lines':[202,231],'observation':'202修改runner加入stage instrumentation；231明确删除stage_profile和sys.setprofile，随后attempt06成功','event_timestamp':'unknown; sequence before attempt06 02:28:35Z'}],'limit':'728秒为整个取消尝试生命周期，不是独立测得的profiler纯开销；最终两次必要fullchecker墙钟426.62/594.02秒，不应视为浪费。'},
 {'priority':4,'finding':'linear整套runtime/OPP迁移被主动停止，最小同架构原包迁移只需10.25秒。','evidence':[{'path':'threehost_linear49/FULL_OPP_TRANSFER_STOPPED.json','time':'2026-10-27T02:23:10.767536+00:00','fields':json.loads((B/'threehost_linear49/FULL_OPP_TRANSFER_STOPPED.json').read_text())},{'path':'threehost_linear49/MINIMAL_MIGRATION_RECEIPT.json','seconds':10.254395961761475,'timestamp':'unknown'},{'path':'/root/dyq/codex-workers/jobs/3591f809e0574104/events.jsonl','line':191,'observation':'migration进程/归档检查单条输出100469字符','timestamp':'unknown'}],'limit':'全套传输起止/纯传输耗时未知，不用最小包时间推算节省量。'},
 {'priority':5,'finding':'同类失败复现与成功补证重跑必须分开。PTO torch.npu缺失4次；NN MatMulV3 500002三次跨host/环境路线。并未证明所有重试环境不变。','evidence':{'pto_same_error':[],'nn_same_error':[r for r in W['threehost_nn11']['attempts'] if 'MatMulV3' in str(r['checker_error'])],'pto_duplicate_successes':[]},'limit':'PTO26成功收据只有23唯一key，初次证据缺失及两次fresh cache重跑有明确补证目的；未发现已accepted成功项无理由重跑。是否每次失败前所有依赖完全不变未知。'}]
for r in W['threehost_pto_original23']['attempts']:
 if r['exit']==1:
  log=B/r['guard'].replace('.json','.log')
  matches=[{'path':str(log.relative_to(B)),'line':i,'excerpt':l.strip()} for i,l in enumerate(log.read_text(errors='replace').splitlines(),1) if "has no attribute 'npu'" in l or 'VERSION file not found' in l]
  if matches:A['findings'][-1]['evidence']['pto_same_error'].append({'start':r['start'],'end':r['end'],'guard':r['guard'],'matches':matches})
counts=collections.Counter(tuple(r['key']) for r in W['threehost_pto_original23']['attempts'] if r['fullcheck_pass'])
A['findings'][-1]['evidence']['pto_duplicate_successes']=[r for r in W['threehost_pto_original23']['attempts'] if r['fullcheck_pass'] and counts[tuple(r['key'])]>1]
A['prompt_corrections']=[
 '每host首次提交前，仅CPU检查准确入口、Python overlay、VERSION/stdlib/deps、metadata路径、架构与原包闭包；先1个最快exactkey完成完整证据链再扩批。涉及设备初始化的检查仍必须guard。',
 '同一错误签名再次提交前，落盘changed_since_previous={文件/环境字段,前后hash,该改变为何消除此错误}；无可验证变化则停止该key并返回阻塞证据。guard75/76仅在新的可用性证据后重试，禁止忙等式同机原样预约。',
 '历史证据仅读给定索引和exactkey字段；禁止整份SCOPE/RESULT/DELIVERY/SDK assembly输出。每命令输出上限4000字符，长日志保存路径+行号+首个错误，远端收据只投影所需字段。',
 '正确性任务禁止自行加全局sys.setprofile、调线程或复制完整runtime/OPP；先复用已验证同host原包，确有缺件只传最小同架构依赖并保留哈希。',
 '一次通过立即冻结checker+guard+原实现来源并提交单key登记包；启动前校验观测器，避免通过后因证据不完整重跑。报告分列fullchecker通过/待登记/accepted，不以CLI completed充数。'
]
pto=W['threehost_pto_original23'];pto['events']['window_commands']=pto['events']['commands'];pto['events']['window_output_characters']=pto['events']['output_characters'];pto['events']['window_count_basis']='entire worker lifecycle is contained in fixed window, despite individual event timestamps unknown'
A['unknowns']=['本次未核查正式登记账本，用户最近10 accepted/h仅为任务上下文，不能用33数值通过替换。','其他三个worker逐命令时间与窗口字符数未知；事件无时间戳。','未量化模型思考、传输纯耗时、设备利用率；attempt墙钟包含编译/reference/等待，不能等价device时间。','报告中环境失败同签名不等价全部环境不变。','一次审计探测对list型RESULT误用list(x)，导致线性任务旧记录输出超限；已改为字段投影，PTO RESULT/DELIVERY始终仅投影字段；未写其他目录。']
A['checks']={'raw_events_workers':4,'raw_guard_receipts':sum(len(w['attempts']) for w in A['workers']),'selected_pto_keys':len(pto['selected_receipts']),'json_parsed':True,'npu_used':False,'cases_rerun':0,'registry_modified':False,'accepted_added':0}
A['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/'AUDIT.json').write_text(json.dumps(A,ensure_ascii=False,indent=2)+'\n')
summary='''最大共性：部署/观测前置检查不足，叠加重复准入等待和过宽读取；不能把低吞吐全归NPU慢。
窗口10-27 02:17:42–03:17:42Z。以下依linear49/attention53/nn11/pto23顺序：总时长3432/3076/3524/2355秒；窗口重叠2430/2074/2521/2355秒，不能相加当墙钟。生命周期命令229/131/150/113，输出518556/959110/1374400/1731124字符（events.jsonl）；前三者窗口命令数未知，PTO全部在窗口。
窗口完整数值通过8/2/0/23唯一key；PTO26份通过收据含3次补证重复。均非正式accepted，本审计accepted_added=0，未核账本，不能取代用户“最近10”口径。
原证据（完整路径见AUDIT）：
1.NN的server1_bf16_02于02:17:54缺einops；remote_bf16_final01/sdk02/graph04于02:29/36/49三次MatMulV3失败。PTO四次torch.npu缺失在02:51–54（guard.log，详AUDIT）。相同错误不证明环境完全不变。
2.attention的remote_evidence/*attempt05/guard.json：02:16:12–02:28:20取消，窗口639秒；events行231删除stage_profile后成功。不是纯profiler耗时。
3.linear的FULL_OPP_TRANSFER_STOPPED.json于02:23:10停止整套传输；最小包10.25秒，全套浪费时长未知。
4.PTO events行59整份SDK收据1048606字符；四任务623命令共4583190字符。路径、时间边界、原log行号见AUDIT。
修正：①每host先CPU精确预检、最快单key闭环再扩批；②同错重试必须记录依赖/环境变化，无变化停止；③仅exactkey字段，单命令输出≤4000字符；④禁止额外全局profiling/调线程/整套runtime复制；⑤通过立即冻结完整证据交登记，事前验证观测器。
已抽核4份events及83份guard与相应checker/log；未初始化NPU、未重跑、未改registry/原实现/记忆。'''
assert len(summary)<=1000,len(summary)
(P/'SUMMARY.txt').write_text(summary+'\n')
for n in ['RESULT.json','DELIVERY.json']:(P/n).write_text(json.dumps({'status':'completed_cpu_evidence_audit','accepted_added':0,'finished_at':A['finished_at'],'findings':'AUDIT.json','summary':'SUMMARY.txt','checks':A['checks'],'remaining_issues':A['unknowns'][:4]},ensure_ascii=False,indent=2)+'\n')
(P/'HEARTBEAT.json').write_text(json.dumps({'phase':'completed','at':A['finished_at'],'accepted_added':0,'npu_used':False},indent=2)+'\n')
files=[p for p in P.iterdir() if p.is_file() and p.name not in ['FILEHASHES.json','SPAWN.json','SCOPE.json','TASK.txt']]
(P/'FILEHASHES.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('summary_chars',len(summary),'checks',A['checks'],'written',[p.name for p in files]+['FILEHASHES.json'])
