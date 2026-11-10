import json,pathlib,subprocess,os,datetime
P=pathlib.Path(__file__).resolve().parent
messages={
'worker01':'优先本scope中 ops-nn BatchNorm validator-none问题和FLA未完整跑key；MoE已证明原API边界的8项复用具体证据，不逐项重复源码搜索。你为本组FLA共性部署问题唯一修复owner；worker03/07只读复用你的recipe，自身key不转移。',
'worker02':'优先SSDChunkState真实剩余补测与InstanceNorm可部署薄接口检查。FP8非M1已确认缩放轴边界，复用具体证据，不盲目换host重跑。你为本组ops-transformer FP8 API边界证据owner；worker06直接复用，避免重复审计。',
'worker03':'优先ops-nn InstanceNorm和ops-transformer DeltaNet可修薄适配/部署缺项；FLA已有原数值失败原条件不重跑。FLA共性部署修复唯一owner=worker01；你准备独立exactkey，与其只读交付复用，不重复修部署。',
'worker04':'优先MoEGroupedGemm resource_gate两项，核三机容量/兼容，已有build直接复用；你为本组ops-transformer GDN共性部署唯一修复owner，worker05复用你的recipe。原GDN数学失败不改原内核，先交剩余可执行key。',
'worker05':'ops-transformer GDN共性部署唯一owner=worker04，你准备独立exactkey和薄入口，复用其recipe避免重复修同一部署；LogSumExp原API sp<=16边界复用具体证据不重放。请尽快写有实际下一步的HEARTBEAT及逐key结果。',
'worker06':'优先ops-math SumFwdOp未完key的实际原完整checker/薄接口。FP8缩放轴原API边界证据owner=worker02，直接复用；Lerp原API边界保留具体源证据，不重复审计。',
'worker07':'优先FLA KDA和ops-math InstanceNorm的可执行薄接口/原完整checker。sgl GDN原DT_DOUBLE不支持不得改原kernel/reference或同条件重跑。FLA共性部署唯一owner=worker01，你做独立key准备并复用其recipe。'}
receipts=[]
for j in json.loads((P/'CHILD_JOBS.json').read_text())['jobs']:
 name=pathlib.Path(j['directory']).name;n=j['native'];tid=None
 with (pathlib.Path(n['directory'])/'events.jsonl').open() as f:
  for line in f:
   e=json.loads(line)
   if e.get('type')=='thread.started':tid=e['thread_id'];break
 msg='协调分工（scope不变）：'+messages[name]+' 每项完成立即保存RESULT；成功需PUBLISH_QUEUE单项完整guard/checker/native/input/all-output/CPU真实路由及具体登记补丁。所有计时未知=null。五分钟预检无执行步骤必须报明确缺项；队列等待不等于失败，不改验收标准。请在下一HEARTBEAT明确确认本分工并以新产出落实。'
 env=os.environ.copy();env['CODEX_HOME']=n['home']['path']
 r=subprocess.run(['codex','queue','--thread',tid,'--message',msg],env=env,capture_output=True,text=True,timeout=30)
 receipts.append({'worker':name,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'thread':tid,'message':msg,'returncode':r.returncode,'stdout':r.stdout[-600:],'stderr':r.stderr[-600:]})
(P/'STEERING_RECEIPTS.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2))
print(json.dumps([{'worker':r['worker'],'returncode':r['returncode'],'stdout':r['stdout']} for r in receipts],ensure_ascii=False))
