"""CPU-only evidence extraction. Does not import any project or accelerator package."""
import ast, collections, datetime, hashlib, json
from pathlib import Path
O=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace'); R=W/'reports/baseline-integration-700-20261026'
S=W/'ascend_baselines/sgl-kernel-npu/python/sgl_kernel_npu/sgl_kernel_npu'
def dump(name,data): (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
scope=json.loads((O/'SCOPE.json').read_text()); hist=json.loads((O/'HISTORICAL_EXACT_ROWS.json').read_text()); bykey={tuple(x['row']['key']):x for x in hist}
files=[S/x for x in ['fla/chunk.py','fla/fused_sigmoid_gating_recurrent.py','fla/fused_sigmoid_gating_recurrent_decode_optimized.py','mem_cache/kv_cache_store.py','attention/sinks_attention.py','attention/decode_attention.py','attention/fia_blockq_attention.py','attention/gqa_share_sparse_attention.py']]
files += [W/'TileOPs/workloads/linear_attention/gdn.py',W/'TileOPs/workloads/attention/gqa/prefill_paged_kv_append.py',W/'TileOPs/workloads/attention/paged_cache_write.py',R/'sgl_gdn_remote38/integration700_sgl_gdn_remote38.py',R/'sgl_gdn_remote38/integration700_sgl_gdn_l2.py']
audit=[]
for p in files:
 text=p.read_text(); tree=ast.parse(text); lines=text.splitlines()
 defs=[{'name':n.name,'line':n.lineno,'signature':ast.unparse(n.args)} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
 needles=('torch.float64','k_c @ live','b_beta = 1.0','K/V dtypes must match','dst_k.dtype ==','softcap','cache_scale')
 audit.append({'path':str(p),'sha256':digest(p),'ast_parse':'passed','top_level_functions':defs,'contract_lines':[{'line':i,'text':l} for i,l in enumerate(lines,1) if any(n in l for n in needles)]})
dump('SOURCE_AUDIT.json',audit)
receipts=[]; rows=[]
for s in scope:
 key=tuple(s['key']); h=bykey[key]['row']; checker=h.get('checker')
 if key[2]=='prefill-narrow-state-bfloat16': checker=str(R/'sgl_gdn_remote38/runs/batch06--GDNFwdOp--prefill-narrow-state-bfloat16/checker.json')
 rec=None
 if isinstance(checker,str) and Path(checker).is_file():
  p=Path(checker); d=json.loads(p.read_text()); rec={'key':list(key),'path':str(p.resolve()),'sha256':digest(p),'receipt':d}; receipts.append(rec)
 refs=[]; command=None
 if key[1]=='DeltaNetFwdOp':
  status='prior_exact_numerical_failure_no_changed_condition'; refs=[str(S/'fla/chunk.py')+':264',str(R/'sgl_gdn_remote38/integration700_sgl_gdn_l2.py')+':36']; change='Root must identify a proven deployment/interface defect causally explaining the exact numerical mismatch, while retaining original implementation/checker; none found. Restoring missing files alone is not a retry condition.'
 elif key[1]=='GDNFwdOp' and key[2]!='decode-gate-fused-bfloat16':
  status='blocked_original_reference_dt_double'; refs=[str(S/'fla/chunk.py')+':53',str(W/'TileOPs/workloads/linear_attention/gdn.py')+':266',str(W/'TileOPs/workloads/linear_attention/gdn.py')+':319']; change='A supported original runtime must demonstrably execute the unchanged float64 reference matmul and solve_triangular. Package version/listing alone is not proof; changing reference precision/device/math is prohibited. Restore exact source and runner dependencies on assigned host before any authorized check.'
 else:
  status='blocked_original_complete_api_contract'
  if key[1]=='GDNFwdOp': refs=[str(S/'fla/chunk.py')+':53',str(S/'fla/fused_sigmoid_gating_recurrent.py')+':120',str(S/'fla/fused_sigmoid_gating_recurrent.py')+':168',str(S/'fla/fused_sigmoid_gating_recurrent_decode_optimized.py')+':96',str(W/'TileOPs/src/tileops/manifest/spec/linear_attention.yaml')+':128']; change='Locate an existing original complete raw-gate API accepting already transformed beta without forced sigmoid; neither inspected recurrent API provides it. No inverse sigmoid or adapter gate arithmetic.'
  else:
   refs=[str(S/'mem_cache/kv_cache_store.py')+':59',str(S/'mem_cache/kv_cache_store.py')+':107']
   if key[1].startswith('GQA'): refs += [str(S/'attention/sinks_attention.py')+':241',str(S/'attention/decode_attention.py')+':373']
   change='Locate an original complete API satisfying '+('FP8 cache quantization/dequantization scales' if 'float8' in key[2] else 'logits softcap')+(' and logits softcap' if 'float8' in key[2] and 'softcap' in key[2] else '')+'. None found in inspected APIs. No added math, store/attention suboperator composition or system fallback.'
 if key[1] in ('GDNFwdOp','DeltaNetFwdOp') and key[2]!='decode-gate-fused-bfloat16':
  command="ssh openlibing-DevEnv-661495 \"npu-guard run --owner 'non_tla_other71/sgl_recurrent_attention/authorized-followup' --wait 600 --timeout 600 -- bash /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/non_tla_other71/sgl_recurrent_attention/device_bundle/baseline-run sgl-kernel-npu python /opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/non_tla_other71/sgl_recurrent_attention/device_bundle/run_case.py "+key[1]+' '+key[2]+' authorized_changed_condition\"'
 rows.append({'key':list(key),'status':status,'historical_status':h.get('status'),'historical_failure_reason':h.get('error',h.get('reason',s['reason'])),'historical_evidence':bykey[key]['source'],'original_interface_refs':refs,'current_source_evidence':'SOURCE_AUDIT.json','current_deployment_evidence':'DEPLOYMENT_AUDIT.json','deployment_finding':'Original provider hardcoded pristine source path and historical worker07 runner absent on assigned host. Overlay package identities are metadata only, not execution proof.','original_checker_receipt':rec['path'] if rec else None,'historical_checker_status':rec['receipt'].get('status') if rec else None,'historical_observed_calls':rec['receipt'].get('observed_calls') if rec else None,'changed_condition_proven':False,'execution_eligible':False,'private_adapter':None,'recipe':'DEPLOYMENT_RECIPE.md','minimal_device_validation':{'command':command,'execute_now':False,'gates':['explicit NPU authorization','documented causal condition change','complete deployment and checker/source pin match'],'wait_seconds':600 if command else None,'run_timeout_seconds':600 if command else None,'budget_note':'One exact fullchecker only; 600s ceiling provisional from prior worker budget, not a performance estimate. API-blocked keys have no valid invocation until complete API exists.'},'required_change':change,'runtime_ready':False,'registration_ready':False,'executed_this_task':False,'device_attempts':[],'accepted_added':0})
dump('CHECKER_RECEIPTS.json',receipts)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
dump('RESULT.json',{'utc':now,'task':'CPU original-API and deployment audit','host':'openlibing-DevEnv-661495','scope_count':len(scope),'status_counts':dict(collections.Counter(r['status'] for r in rows)),'rows':rows,'device_attempts':0,'accepted_added':0,'publish_candidates':0,'checks':{'stdlib_only_source_ast_files':len(audit),'exact_checker_receipts':len(receipts),'remote_readonly_inventory':True,'project_imports':0,'npu_imports':0},'timings':{'device_seconds':0,'queue_seconds':0,'cpu_seconds':None},'limitations':['No runtime import or device validation authorized.','Missing fixed-host deployment does not explain or repair prior oracle/numerical/API failures.','No valid passed original checker found in scoped evidence.']})
dump('PUBLISH_CANDIDATES.json',{'utc':now,'scope_count':13,'candidates':[],'reason':'Seven scoped original checker receipts report failure; remaining six scoped API-blocked keys have no passing checker evidence. No reruns.'})
dump('HEARTBEAT.json',{'utc':now,'phase':'completed','keys':13,'npu_attempts':0,'last_output':'RESULT.json; source/deployment/checker evidence; recipe','blockers':'6 reference DT_DOUBLE; 6 API contracts; 1 numerical failure; assigned-host deployment missing'})
assert len(rows)==13 and len({tuple(r['key']) for r in rows})==13
assert len(receipts)==7 and all(r['receipt']['status']=='failed' for r in receipts)
print(json.dumps({'scope':len(rows),'receipts':len(receipts),'ast_files':len(audit),'counts':collections.Counter(r['status'] for r in rows)}))
