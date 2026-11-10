import pathlib,json,hashlib,datetime
P=pathlib.Path(__file__).resolve().parent;W=P.parents[2]
files={
'sgl_default_arch':('ascend_baselines/sgl-kernel-npu/csrc/CMakeLists.txt',[(15,38),(88,107),(150,163)]),
'sgl_store':('ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/mem_cache/kv_cache_store.py',[(9,73),(105,113)]),
'sgl_sinks':('ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py',[(139,170),(241,285)]),
'moe_normal':('ascend_baselines/sgl-kernel-npu/python/deep_ep/deep_ep/strategies/normal_strategy.py',[(595,626),(679,727),(751,831)]),
'moe_ep':('ascend_baselines/sgl-kernel-npu/csrc/deepep/ops/op_host/moe_distribute_combine_v2_tiling.cpp',[(88,96),(354,370)]),
'flash_api':('ascend_baselines/ops-transformer/attention/flash_attn/torch_extension/flash_attn.py',[(360,422)]),
'quant_dtype':('ascend_baselines/ops-transformer/attention/quant_flash_attn/op_host/quant_flash_attn_infershape.cpp',[(205,219)]),
'cumsum_vector':('ascend_baselines/vllm-ascend/vllm_ascend/ops/triton/kda/cumsum.py',[(122,158),(194,232)]),
'gdn_reference':('TileOPs/workloads/linear_attention/gdn.py',[(260,277),(310,350)]),
'kda_reference':('TileOPs/workloads/linear_attention/kda.py',[(163,181),(229,248)]),
'fullchecker_reference_device':('TileOPs/scripts/ascend_pair_check.py',[(345,389)]),
'sum_binding':('TileOPs/benchmarks/ascend_all_pairs/ops_tensor/reductions.py',[(1,160)])}
records={}
for name,(rel,ranges) in files.items():
 p=W/rel
 if not p.exists():records[name]={'missing':str(p)};continue
 data=p.read_bytes();lines=data.decode().splitlines();records[name]={'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'excerpts':[{'line':n,'text':lines[n-1]} for a,b in ranges for n in range(a,min(b,len(lines))+1)]}
(P/'CURRENT_SOURCE_AUDIT.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
scope=json.load(open(P/'SCOPE.json')); rows=[]
for i,r in enumerate(scope):
 row={'key':{k:r[k] for k in ['baseline','operator','case_id']},'status':'pending','prior_status':r.get('status'),'prior_reason':r.get('reason'),'accepted_added':0,'registered':False,'numerical_success':False,'executed_this_task':False,'guard_receipt':None,'fullchecker_receipt':None,'native_evidence':None,'timings':{'queue_seconds':None,'physical_idle_wait_seconds':None,'device_command_seconds':None},'checked_outputs':0,'total_outputs':r.get('total_outputs'),'audit_evidence':[]}
 if i<15 or i in (27,28):
  row.update(status='blocked_deployment',reason='Current three-host inspection: local topology package absent; both remote /etc/hccl_rootinfo.json are read-only mounted empty directories. No authoritative real topology available inside allowed write scope; no fabricated topology or communication bypass. Original alltoall remains communication dependent.',audit_evidence=['CURRENT_SOURCE_AUDIT.json#moe_normal','CURRENT_SOURCE_AUDIT.json#moe_ep','local-topology.json','openlibing-DevEnv-661495-topology.json','openlibing-DevEnv-055409-topology.json'])
 elif i in (16,17):row.update(status='validation_in_progress' if i==17 else 'pending_budgeted_validation',reason='Original fullchecker and provider frozen; extended runtime budget authorized. No numerical result yet.')
 else:
  row.update(status='blocked_original_contract_or_environment',reason=r.get('reason'),audit_evidence=['CURRENT_SOURCE_AUDIT.json'])
  if i==26:row['audit_evidence'].append('../mla_original_api_audit1/CONTRACT_OR_BUILD_EVIDENCE.json')
 rows.append(row)
(P/'RESULT.json').write_text(json.dumps({'scope_total':len(rows),'accepted_added':0,'rows':rows,'in_progress':True},ensure_ascii=False,indent=2))
h={'phase':'cpu_contract_audit_complete_dsa96_running','currentkey':scope[17]['case_id'],'host':'server2','lastprogress':datetime.datetime.now(datetime.timezone.utc).isoformat(),'blockedreason':None};(P/'HEARTBEAT.json').write_text(json.dumps(h))
print(len(rows),'rows',len(records),'source audits')
