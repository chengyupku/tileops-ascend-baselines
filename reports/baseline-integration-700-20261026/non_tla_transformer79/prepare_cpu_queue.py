import json,pathlib,fcntl,datetime
D=pathlib.Path('/root/dyq/workspace/reports/baseline-integration-700-20261026');G=D/'non_tla_transformer79'
def read(p):return json.loads(p.read_text())
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
s=read(G/'SCOPE.json'); families={
 'fp8_original15':{'GemmFP8FwdOp'},
 'moe_original8':{'FusedMoEExpertsFwdOp'},
 'linear_original16':{'GDNFwdOp','DeltaNetFwdOp','SSDChunkStateFwdOp'},
 'attention_original20':{'GQAPagedFwdOp','GQAVarlenFwdOp','GQADenseFwdOp','GQAPrefillPagedWithKVCacheFwdOp','MLAPagedFwdOp','NSACompressedVarlenFwdOp','NSAVarlenFwdOp','PagedKVCacheWriteFwdOp'},
 'reduction_elementwise20':{'AmaxFwdOp','ClampScalarFwdOp','LogSumExpFwdOp','LerpTensorFwdOp','InstanceNormFwdOp'}}
pre=read(G/'OWNERSHIP_PREFLIGHT.json');assert not any(r['child_live'] or r['status'] in ('starting','running','stopping') for r in pre['old_native']);assert all(r['rc']==0 and r['status']['state']=='idle' and not r['status']['reservation']['locked'] for r in pre['remote_guard'])
queue=[]; covered=[]
for name,ops in families.items():
 rows=[r for r in s if r['key'][1] in ops];q=G/name;q.mkdir(exist_ok=True);put(q/'SCOPE.json',rows);covered+=rows
 put(q/'HOST_ASSIGNMENT.json',{'host':'openlibing-DevEnv-055409','scope_keys':sorted(r['key'] for r in rows),'approved_by_coordinator':True,'allow_npu':False,'phase':'CPU original API and deployment verification only; host pinned for read-only remote filesystem checks'})
 task=f'''你是原baseline顶层接口/部署核实执行worker。唯一范围 {q}/SCOPE.json，共{len(rows)}个exactkey，可写仅{q}。禁止NPU，包括任何初始化设备的import；禁止派遣；本机与远端均不得运行checker/benchmark。可ssh openlibing-DevEnv-055409做纯文件/命令存在性/原包metadata检查；不可自行换host。CPU准备不持设备锁。
先读AGENTS.md、MEMORY索引、项目BASELINE_RETEST_OPERATING_RULES_20261027.md和BASELINE_WORKER_PROMPT_RULES_20261027.txt。仅按SCOPE.evidence定位历史，禁止全读巨大STATE/SNAPSHOTS。当前只暂停tilelang-ascend，tilelang不同，但本任务范围均ops-transformer。
任务：针对{sorted(ops)}核对当前原源码完整API、已部署原库、现有顶层provider与checker contract，找能恢复一批key的入口/参数/包装/部署问题。历史标签不当永久结论，但原数值失败/API不支持不得无条件重试。原kernel/compiler/pass/算法/数学/reference/checker/seed/tolerance不得改；不得拼接子算子实现缺失API。只允许在私有目录写语义等价顶层适配候选及CPU静态验证，不写共享providers/ACCEPTED/其它任务。
先排查已成功待发布证据；若存在完整成功，复用而不重跑，提供具体checker/native/capture/registration_pins路径和原身份。否则逐key给出原公开完整API存在性、签名/能力、现有封装差异、当前部署验证证据、历史是否真实数值失败、可验证changed_condition、下一条明确合法动作。对于可接项交私有薄适配、部署recipe、原checker命令、所需guard预算给root，禁止自己执行设备。不能仅复述历史报告。同族公共API一次审查，各exact contract明确映射，避免无意义重复。
5分钟内输出首个结论或具体缺件，阶段变更写HEARTBEAT.json。完成写RESULT.json含rows逐key证据与source路径/行号；状态严格区分cpu_ready_for_guarded_check / complete_api_unsupported / original_numerical_failure / deployment_blocked / verified_pending_publication（最后者必须真实完整checker）。正式发布只能交acceptance_root，不能自称accepted。完成通知由native supervisor自动发送，不自行queue任何人。
'''
 (q/'TASK.txt').write_text(task)
 queue.append({'name':name,'directory':str(q),'ready':True,'approved_by_coordinator':True,'allow_npu':False,'retry':False,'changed_condition':'Fresh current original API and deployment CPU verification, no numerical retry','previous_devices_quiescent':True})
assert len(covered)==79 and len({tuple(r['key']) for r in covered})==79
with (G/'CHILD_DISPATCH.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);put(G/'READY_QUEUE.json',{'tasks':queue});group=read(G/'GROUP.json');group.update(enabled=True,phase='running',cpu_only=True,ownership_preflight=str(G/'OWNERSHIP_PREFLIGHT.json'));put(G/'GROUP.json',group)
put(G/'HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_delta':0,'ready':len(queue),'running':0,'npu_queue':0,'next_action':'refill service dispatch 5 disjoint CPU original API/deployment workers','scope_count':79})
print([(x['name'],len(read(pathlib.Path(x['directory'])/'SCOPE.json'))) for x in queue])
