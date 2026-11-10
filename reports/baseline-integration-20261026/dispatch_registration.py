import pathlib,json,subprocess,datetime
D=pathlib.Path(__file__).resolve().parent;summary=json.load(open(D/'REVIEWED_RESULTS.json'));defs=[('local',['nn_norm','nn_misc_math/recovery01'],['ops_nn']),('server1',['tla_direct'],['tla']),('server2',['transformer_attention','transformer_gdn'],['ops_transformer'])];jobs=[]
for host,groups,providers in defs:
 R=D/('register_'+host);R.mkdir(exist_ok=True)
 if (R/'JOB.json').exists():raise SystemExit('already submitted; no replay')
 selected=[x for x in summary['rows'] if x['status']=='verified' and x['group'] in groups];(R/'SCOPE.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
 project='/root/dyq/workspace' if host=='local' else '/opt/tileops-all-pairs/workspace';remote=project+'/reports/baseline-integration-20261026/register_'+host
 task=f'''用户已授权实际baseline对接；主agent已核验完成数值验证的56个exact key。现在完成其中{len(selected)}个的正式调用入口整合，host={host}，SCOPE={R}/SCOPE.json。不是重新分类，也不是重新运行数值验证。本任务明确授权写下面列明的共享provider目录，取代此前叶子任务“共享只读”的限制；其他限制不变。
目标：把已验证的私有完整原版API适配，连同必要的host环境/运行配方，接入现有baseline正常调用路径。用户不应只能在临时run_case.py的monkeypatch下使用。复用项目原有注册/部署方式；确认真正可调用的factory及环境入口。只登记SCOPE内已通过exact keys，保留其他现有provider/accepted keys，未知case继续原逻辑。禁止从分类high标签扩充成功。
当前已验证来源group={groups}；读这些目录的DELIVERY.json和PARENT_REVIEW.json，引用其完整checker与guard receipts，不再跑已通过case。已有全部原始日志。若迁移后改变了数学、dtype/layout逻辑或runtime，不能继承通过结论；应保持被测实现不变。
本机可写={R}。执行host项目根={project}。执行host可写任务目录={remote}，以及共享provider目录：{[project+'/TileOPs/benchmarks/ascend_all_pairs/'+x for x in providers]}。必要新模块放在自己所属provider目录内；禁止改公共registry.py、公共GraphCall、其他provider、vendor/kernel/compiler、工作负载oracle/verifier、全局环境、旧HOLD/调度器、记忆或其他任务文件。修改前逐文件备份到本任务目录并记hash。最终给出精确patch、CHANGED_FILES.json、注册key到factory与既有数值收据映射、deployment recipe、验证结果。
只做CPU静态核查与device-free注册测试，明确禁止NPU初始化/运行/数值重跑/benchmark。不得import会初始化torch_npu的模块；AST解析、compile()、受控mock registry测试可以使用。不用新建镜像/重装包，不做大规模拷贝。已有私有runtime和source paths可通过清晰的配置引用保留，不要改造成未知部署。不得为了静态测试导入真正设备模块。
特别注意：nn_norm的adapter.patch会宽泛影响InstanceNorm，应只为3个已通过GroupNorm key选择对应已验证路径，2个失败InstanceNorm不能标已通过。nn_misc的Sum失败不能注册通过。transformer_attention的示意patch会把公共GraphCall所有库重定向，禁止直接应用；用provider内部独立helper/明确vendor路径保持原行为。GDN必须保持v-first状态合同和2个输出，不注册4个DeltaNet或6个资源门槛未过的key。TLA Where只此256M-float16，保留原版PTO、BLOCK_M/BLOCK_N、bitpack；其余5个未通过不得启用。
三个整合agent各自拥有不同host/provider，不会互写。不要再派子agent。全部medium，不能换模型。原生监督器自动queue完成通知。禁止启动长期监控。
验收：正式入口覆盖所有分配已通过keys，现有keys不丢，静态检查通过、实现/依赖与原被测路径一致有hash/diff依据，原成功checker无需重跑。如需写范围外文件，先在本任务目录形成精确patch并报告阻塞，不能自行越界。完成报告明确registered_count、verified_receipts_reused_count、任何尚未整合key，不把静态检查称为新数值成功。
'''
 if host!='local':
  alias={'server1':'openlibing-DevEnv-661495','server2':'openlibing-DevEnv-055409'}[host]
  task+=f'远端用 /root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh {alias}。本地其他报告只读；远端证据和patch复制回{R}。只在此host改共享provider；不要把远端二进制装到本机。\n'
 (R/'TASK.txt').write_text(task)
 args=['python3','/root/dyq/workspace/.cache/all-pairs/full_rollout_20261024/codex_worker_manual_quota.py','--native-script','/root/dyq/tools/codex-worker.py','spawn','--cwd','/root/dyq/workspace','--task-file',str(R/'TASK.txt'),'--write',str(R)]
 if host=='local':
  for x in providers:args+=['--write',project+'/TileOPs/benchmarks/ascend_all_pairs/'+x]
 args+=['--model','gpt-6-astra','--effort','medium','--timeout','7200','--notify','queue','--sandbox','danger-full-access']
 r=subprocess.run(args,text=True,capture_output=True,timeout=45);v={'host':host,'count':len(selected),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 try:v['native']=json.loads(r.stdout)
 except ValueError:pass
 (R/'JOB.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));jobs.append(v);(D/'REGISTRATION_JOBS.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2))
 print(json.dumps({'host':host,'count':len(selected),'returncode':r.returncode,'id':v.get('native',{}).get('id'),'stderr':r.stderr}),flush=True)
 if r.returncode:break
