import json, hashlib, re
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path('/root/dyq/workspace')
OUT=ROOT/'reports/baseline-integration-700-20261026/fla_original_precision_audit14'
OLD=OUT.parent/'fla'
BASE='ascend_baselines/flash-linear-attention-npu/'
GDN=BASE+'fla/ops/ascendc/gdn/'
CH=GDN+'chunk_gdn_fwd/chunk_gated_delta_rule_fwd/'
REC=GDN+'recurrent_gdn/recurrent_gated_delta_rule/'
KDA=BASE+'fla/ops/ascendc/kda/'
now=datetime.now(timezone.utc).isoformat()
scope=json.loads((OUT/'SCOPE.json').read_text())
old=json.loads((OLD/'RESULT.json').read_text())['rows']
files={
 'chunk_api':(CH+'op_host/op_api/aclnn_chunk_gated_delta_rule_fwd.cpp',[(103,131),(553,593),(669,676),(686,725),(743,768)]),
 'chunk_tiling':(CH+'op_host/op_tiling/arch35/chunk_gated_delta_rule_fwd_arch35_tiling.cpp',[(97,113),(159,203),(229,237),(338,341)]),
 'chunk_kernel':(CH+'op_kernel/internal/arch35/chunk_gated_delta_rule_fwd_arch35.cpp',[(416,437),(480,505)]),
 'recurrent_api':(REC+'op_host/op_api/aclnn_recurrent_gated_delta_rule.cpp',[(64,73),(89,111)]),
 'recurrent_accumulator':(REC+'op_kernel/recurrent_gated_delta_rule.h',[(390,440)]),
 'stable':(BASE+'torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py',[(511,566),(1488,1538)]),
 'beta_binding':(BASE+'torch_custom/fla_npu/csrc/src/stable_chunk_gated_delta_rule_fwd.cpp',[(95,135),(156,180)]),
 'recurrent_kda_api':(KDA+'recurrent_kda/op_host/op_api/aclnn_recurrent_kda.cpp',[(72,78),(396,405)]),
 'chunk_kda_api':(KDA+'chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp',[(340,365),(388,390),(579,590)]),
 'public_flash':(BASE+'examples/flash_gated_delta_rule.py',[(1108,1116),(1162,1207)]),
 'manifest':('TileOPs/src/tileops/manifest/spec/linear_attention.yaml',[(40,48),(110,130),(252,260)]),
 'gdn_contract':('TileOPs/workloads/linear_attention/gdn.py',[(167,190),(252,266),(292,306),(330,335),(352,390)]),
 'delta_contract':('TileOPs/workloads/linear_attention/deltanet.py',[(415,430)]),
 'rounding':('TileOPs/workloads/numerics.py',[(586,597)]),
 'seed':('tools/baseline_protocol.py',[(47,50)]),
 'adapter':('reports/baseline-integration-700-20261026/fla/integration700_fla_native.py',[(43,68)]),
 'checker':('TileOPs/scripts/ascend_pair_check.py',[(235,242)]),
 'checker_difference':('reports/baseline-integration-700-20261026/fla/reused_checker.diff',[(1,34)])}
evidence={}
for key,(path,ranges) in files.items():
 p=ROOT/path; raw=p.read_bytes(); lines=raw.decode().splitlines(); digest=hashlib.sha256(raw).hexdigest()
 pins={r['case_id']:r.get('source_hash',{}).get(path) for r in scope if path in r.get('source_hash',{})}
 evidence[key]={'path':str(p),'sha256':digest,'historical_scope_pins':pins,'historical_pin_match':{k:v==digest for k,v in pins.items()},'excerpts':[{'start_line':a,'end_line':min(b,len(lines)),'text':'\n'.join(f'{i}: {lines[i-1]}' for i in range(a,min(b,len(lines))+1))} for a,b in ranges]}
manifest=(ROOT/files['manifest'][0]).read_text().splitlines()
rows=[]
for s in scope:
 cid,op=s['case_id'],s['operator']; unrun=s.get('evidence_scope')=='representative_only_not_exact_key_execution'; kda=op=='KDAFwdOp'
 seed=int.from_bytes(hashlib.sha256(f'{op}::{cid}'.encode()).digest()[:4],'big')
 row={'baseline':s['baseline'],'operator':op,'case_id':cid,'status':s['status'],'prior_execution':'not_executed_exact_key' if unrun else ('oracle_blocked_before_candidate' if kda else 'numeric_failed_exact_key'),'actionability':'no','actionability_definition':'New legal precision/dispatch repair supported enough to recommend a parent NPU probe; not a prediction that unchanged unrun keys fail.','accepted_added':0,'registered':False,'factory':None,'checked_outputs_this_audit':0,'total_outputs':2,'api':s['api'],'checker_receipt':s.get('checker_receipt'),'guard_receipt':None,'timings':{'queue':0,'device':0,'compile':0},'seed':seed,'seed_evidence':'seed:47-50 (hash-only computation, no torch)','changed_files':[]}
 if kda:
  row.update(reason='Carry forward existing FP64 NPU oracle blocker and pending proposal unchanged; no new oracle investigation or candidate run.',new_evidence='No new KDA oracle evidence requested; existing PROPOSED_KDA_checker.patch retained read-only.',pending_proposal=str(OLD/'PROPOSED_KDA_checker.patch'),source_refs=[],confirmed=['Existing scope says candidate never reached.'],inference=['No inference about candidate numerical correctness.'])
 else:
  label=cid.rsplit('-',1)[0]; matches=[(i+1,t) for i,t in enumerate(manifest) if 'label: '+label+'}' in t or 'label: '+label+',' in t]
  assert len(matches)==1,(cid,matches)
  ln,line=matches[0]; dims={name:int(re.search(r'\b'+name+r': (\d+)',line).group(1)) for name in ['B','S','H','K','V']}; dims['HV']=int(re.search(r'\bHV: (\d+)',line).group(1)) if 'HV:' in line else dims['H']
  l2='beta-l2norm' in cid; delta=op=='DeltaNetFwdOp'
  row['manifest']={'path':str(ROOT/files['manifest'][0]),'line':ln,'text':line,'dimensions':dims}
  row['contract']={'qkv_dtype':'bfloat16' if l2 else 'float16','gate_beta_caller_dtype':'bfloat16' if l2 else 'float16','adapter_gate_beta_dtype':'float32','initial_state_dtype':'float32','output_dtype':'bfloat16' if l2 else 'float16','final_state_dtype':'float32','scale':'None -> K**-0.5 (128**-0.5), unchanged','l2norm':l2,'beta_sigmoid':l2,'allow_neg_eigval':l2,'raw_gate':False,'state_v_first':False,'output_comparison':'assert_rounded (one adjacent representable value) then absolute/relative bound','output_atol':4e-8 if l2 or delta else 6e-7,'output_rtol':0 if delta else (4e-8 if l2 else 6e-7),'state_atol':4e-8 if l2 or delta else 6e-7,'state_rtol':4e-8 if l2 or delta else 6e-7,'oracle':'existing single-token FP32 recurrence; unchanged','controls':'query-zeroed and gate-zeroed' if not delta else 'existing Delta Custom verifier'}
  row['source_refs']=['chunk_api','chunk_tiling','chunk_kernel','recurrent_api','stable','public_flash','recurrent_kda_api','chunk_kda_api','adapter','manifest','gdn_contract' if not delta else 'delta_contract','seed','rounding']
  row['confirmed']=['Full chunk GDN rejects FP32 q/k/v before dispatch (chunk_api 553-559); FP32 gate/beta/state support is not FP32 QKV support.','Full recurrent GDN only admits BF16 QKV/beta, not FP16 or FP32 (recurrent_api 64-73,89-111); FP16->BF16 is not lossless widening.','Full recurrent KDA only admits BF16 QKV; chunk KDA admits FP16/BF16 but rejects FP32 QKV.','Public flash_gated_delta_rule rejects FP32 explicitly; original output cast exists at 1115 but does not create FP32-input support.','Existing adapter already widens gate/beta losslessly, clones FP32 state, preserves scale and requests both outputs.']
  if l2:
   row['dispatch']='BF16 K=V=128, Hq=Hv=4, chunk64: Ascend950 UsePreparePath=true; beta sigmoid/negative eigenvalue and QK L2 handled inside original full API.'
   row['confirmed']+=['This is a different original path from the FP16 representatives: prepare -> ChunkFwdH -> ChunkFwdO. Binding allocates beta_eff only when sigmoid requested (beta_binding 118-122); original prepare receives normalization/negative-eigenvalue flags (chunk_api 746-752).','Prepared qHat/kHat/w/u/h/vNew remain input dtype while rstd/beta_eff are FP32 (chunk_api 697-723). Native recurrent GDN has no L2/sigmoid arguments and cannot replace this case without forbidden external math.']
   row['source_refs']+=['beta_binding']
   row['reason']='Original BF16 prepare path already implements flags but rounds normalized Q/K/intermediates to BF16; full API rejects QKV FP32 widening. No proven alternative FP32 full API; no external normalization/sigmoid repair allowed.'
  else:
   row['dispatch']='Ascend950 FP16 BNSD dense S=1 K=V=128: UsePreparePath=false; Phase6 V128 tiling key 1; B30 false (requires BF16 + varlen).'
   row['confirmed']+=['Batch '+str(dims['B'])+' and Hq/Hv='+str(dims['H'])+'/'+str(dims['HV'])+' affect workload/tiling sizes but not UsePreparePath or V128/B30 selector. No distinct FP32 path is selected for this exact shape.','Original tiler configures FP32 matmul result (chunk_tiling 97-113), yet input/A dtype is FP16 and kernel uses typed intermediates; accumulator precision alone is not a repair.']
   row['reason']='No legal FP32 QKV widening route: full chunk API rejects it; recurrent APIs require BF16 QKV. Exact '+str(dims['B'])+'-batch shape selects same Phase6 V128 top-level path, not a new dtype-capable branch.'
   if delta: row['confirmed']+=['Delta adds zero gate through existing adapter; its strict 4e-8 output/state contract differs from ordinary GDN 6e-7. Passing GDN prefill does not establish this decode contract.']
   if unrun: row['reason']+=' This exact key remains unexecuted, not numerically failed.'
  row['inference']=['No recommendation to replay unchanged native route. Shared dtype rejection is proven; shared numerical failure across seeds/batches is NOT proven.','Potential contribution of low-precision intermediates to historical errors is a hypothesis, not an isolated causal measurement.']
  if not unrun:
   receipt=Path(s['checker_receipt']); receipt=receipt if receipt.is_absolute() else ROOT/receipt
   r=json.loads(receipt.read_text()); assert r['input_seed']==seed
   row['historical_receipt']={'path':str(receipt),'sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'input_seed':r['input_seed'],'case_snapshot':r.get('case_snapshot'),'verifier_sha256':r.get('verifier_sha256')}
  else: row['seed_note']='Canonical seed derived from exact key; representative receipt seed differs. No exact-key output or failure is available.'
 rows.append(row)
checks={'cpu_only':True,'real_torch_imports':0,'npu_calls':0,'compiles':0,'numeric_trials':0,'subagents':0,'scope_keys':len(rows),'exact_numeric_failures_retained':sum(r['prior_execution']=='numeric_failed_exact_key' for r in rows),'exact_unrun_retained':sum(r['prior_execution']=='not_executed_exact_key' for r in rows),'oracle_blocked_retained':sum(r['prior_execution']=='oracle_blocked_before_candidate' for r in rows),'historical_seed_checks':4,'manifest_exact_key_checks':13,'source_hashes':len(evidence),'json_scope_validation':'14 unique exact keys; 9 unrun / 4 numeric failures / 1 oracle blocker; accepted_added=0 and registered=false' }
assert checks['scope_keys']==14 and checks['exact_unrun_retained']==9 and checks['exact_numeric_failures_retained']==4
assert all('q/k/v must be float16 or bfloat16' in (ROOT/files['chunk_api'][0]).read_text() for _ in [0])
passed=[{'operator':r['operator'],'case_id':r['case_id'],'api':r['api'],'checker_receipt':r.get('checker_receipt')} for r in old if r['status']=='completed']
E={'created_at':now,'baseline_git_head':'7e39f9668eed285fc91cde0502f46cd48686a681','sources':evidence,'existing_passed_comparison':passed,'comparison_conclusion':'Same baseline has successful native low-precision GDN prefill and zero-state first-token GLA, plus unrelated FP32 Add. None of these receipts establishes a full GDN/Delta FP32-QKV route. Prefill uses looser tolerance and CPU original FP64 reference; single-token decode retains original FP32 reference. Do not transfer those passes to decode or to FP32 widening.','checks':checks,'historical_adapter_caveat':'Current adapter hash differs from decode-b1 and Delta decode historical snapshots, but matches the later beta/L2 and GVA failure pins. Source dtype/dispatch assertions apply to the pinned current sources; no claim of a numerical recheck or historical binary equivalence.', 'limitations':['Current source is pinned; matching historical pins are reported per file, and unpinned files are not claimed identical to historical binaries.','Static source selection does not establish runtime accuracy or compiled binary equivalence.','No broad library/history search or remote investigation performed.']}
R={'created_at':now,'task':'fla_original_precision_audit14','accepted_added':0,'actionable_yes':0,'actionable_no':14,'rows':rows,'checks':checks}
D={'created_at':now,'accepted_added':0,'summary':'No supported lossless QKV FP32 widening route found in original complete GDN/KDA recurrent/chunk APIs. 9 exact unrun keys remain unrun; 4 actual failures retained; KDA oracle proposal carried forward.','first_parent_npu_case':None,'parent_next_step':'Do not schedule an unchanged NPU replay from this audit. Needed evidence is an existing original complete API admitting FP32 QKV with original fused beta/L2 and output/state contract, plus same-baseline full-API success. Current inspected entrypoints explicitly reject FP32 QKV. Existing KDA oracle proposal remains separately pending.','private_adapter_proposal':None,'adapter_omission_reason':'No actionable precision route; a widening adapter would be rejected before kernel dispatch, so no unsupported adapter or misleading stub is supplied.','checks':checks,'reproduce_cpu_only':'python3 '+str(OUT/'build_report.py'),'changed_files':[str(OUT/n) for n in ['build_report.py','DELIVERY.json','RESULT.json','EVIDENCE.json']],'remaining_issues':['Exact-key accuracy for 9 unrun cases unknown; representative failures are not failures of those keys.','No approved new precision repair for the four actual failures.','KDA FP64 oracle pending proposal unchanged.'],'source_hashes':{v['path']:v['sha256'] for v in evidence.values()},'writes_outside_task':False,'registered':False}
for name,obj in [('EVIDENCE.json',E),('RESULT.json',R),('DELIVERY.json',D)]: (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(checks))
