import json,fcntl,datetime,collections,sqlite3,os
from pathlib import Path
D=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026');P=D/'non_tla_other71'
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=json.loads((P/'SCOPE.json').read_text()); groups=collections.defaultdict(list)
for r in s:
 b,op,c=r['key']
 if b=='tilelang' and op=='BmmFwdOp':continue
 g='original_ops' if b.startswith('ops-') else 'fla_recurrent' if b=='flash-linear-attention-npu' else 'sgl_moe_comm' if b=='DeepEP-Ascend' or (b=='sgl-kernel-npu' and ('Permute' in op or 'Gather' in op)) else 'sgl_recurrent_attention' if b=='sgl-kernel-npu' else 'vllm_tilelang'
 groups[g].append(r)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (P/'CHILD_DISPATCH.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);queue=[]
 for name,rows in groups.items():
  q=P/name;q.mkdir(exist_ok=True);put(q/'SCOPE.json',rows)
  put(q/'HOST_ASSIGNMENT.json',{'host':'openlibing-DevEnv-661495','scope_keys':sorted(r['key'] for r in rows),'approved_by_coordinator':True,'allow_npu':False,'phase':'CPU original-interface verification only','local_forbidden':True})
  task=f'''你是原baseline顶层接口CPU适配/部署核实worker。独占范围见 {q}/SCOPE.json，共{len(rows)}项；只写{q}。先读AGENTS.md、记忆索引与项目INDEX、BASELINE_RETEST_OPERATING_RULES_20261027.md及reports/baseline-integration-700-20261026/BASELINE_WORKER_PROMPT_RULES_20261027.txt。禁止整读历史STATE/SNAPSHOTS。不派遣agent、不改记忆。NPU绝对禁止，包括会初始化设备的import；只可读源码、静态检查、ssh只读文件/包部署清单。固定设备host为openlibing-DevEnv-661495，但本任务未获设备运行权，不能自选机或更改远端文件。
任务是{ name }原接口接入准备，不是算子开发。优先可恢复一批key的原依赖、入口、包装问题，以及已成功待发布证据。逐项读取scope引用的有限证据，再直接查看当前原API/源码与实际部署。历史失败标签不可直接当结论；原数值失败/API不支持无明确条件变化禁止重跑。保持原kernel/compiler/pass/math/checker/reference/seed/tolerance。只允许原完整API签名/输入输出封装及部署recipe；禁止子算子拼接或fallback冒充原API。若能修复，私有适配文件放本目录并做不初始化设备的静态检查。无需NPU可完成的材料直接完成。
5分钟内形成具体可执行步骤或缺项，每2分钟HEARTBEAT。交RESULT.json含逐key status、原接口文件行号、当前部署证据、历史失败原因、新条件是否真实满足、私有适配/recipe、最小设备验证命令及预算（只写不执行）、阻塞所需具体改变。若发现历史原checker已通过，收齐原证据生成PUBLISH_CANDIDATES.json，不重跑，不直接写ACCEPTED/providers。不得修改其它root或调度器。完成通知仅由native supervisor自动投递真实父session，无需自己queue。'''
  (q/'TASK.txt').write_text(task)
  queue.append({'directory':str(q),'ready':True,'approved_by_coordinator':True,'allow_npu':False,'retry':False,'purpose':'CPU current original API and deployment verification; no numeric replay'})
 put(P/'READY_QUEUE.json',{'tasks':queue})
 group=json.loads((P/'GROUP.json').read_text());group.update(enabled=True,phase='running',execution_authorized=False,cpu_only=True);put(P/'GROUP.json',group)
 put(P/'HEARTBEAT.json',{'at':now,'accepted_delta':0,'ready':len(queue),'running':0,'device_queue':0,'next_action':'refill launches five mutually exclusive CPU families; root reviews historical Bmm success policy','no_npu_authorized':True})
 put(P/'OWNERSHIP_PREFLIGHT.json',{'at':now,'scope_count':71,'other_root_overlap':0,'native_active_jobs_at_check':['9abab0f327f749b1','37016ef4795d41b3'],'local_handoff_scope':[],'local_handoff_authorized':False,'remote_guard_sample':{'hosts':['openlibing-DevEnv-661495','openlibing-DevEnv-055409'],'state':'idle','locked':False,'driver_pids':[],'fd_candidates':[],'health':0,'errors':[]},'device_execution_authorized':False,'cpu_scope_count':68,'historical_Bmm_passes_held_for_root_review':3,'stale_PLAN_active_assignment_not_native_liveness':True})
 print('queued',[(k,len(v)) for k,v in groups.items()])
