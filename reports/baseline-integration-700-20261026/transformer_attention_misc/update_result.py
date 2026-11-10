import pathlib,json,datetime
P=pathlib.Path(__file__).resolve().parent
expected=json.load(open(P/'EXPECTED_OUTPUTS.json'))
scope=json.load(open(P/'SCOPE.json'));reg={tuple(x) for x in json.load(open(P/'REGISTRATION.json'))['keys']} if (P/'REGISTRATION.json').exists() else set();rows=[]
if (P/'REGISTRATION_GROUPED.json').exists():reg.update(tuple(x) for x in json.load(open(P/'REGISTRATION_GROUPED.json'))['keys'])
block_ops={'GQAPagedFwdOp':'Current registered FlashAttentionScore decode entry lacks the required paged sink/window/RoPE/softcap contract; newer FlashAttn/QuantFlashAttn source candidates have no built extension in inspected server2 source closure. No extra activation/rotation composition introduced.', 'GQAVarlenFwdOp':'Original FlashAttn (sinks) / QuantFlashAttn (FP8 scale) candidate requires its own complete native package. Server2 inspected source has no built torch extension; current FlashAttentionScore package is not equivalent for these flags.', 'GQAPrefillPagedWithKVCacheFwdOp':'Complete paged FlashAttn API requires undeployed native package and metadata extension. No source-owned built closure found in inspected server2 source; existing registered single-query FlashAttentionScore does not implement this full prefill binding.', 'NSACompressedVarlenFwdOp':'Complete FlashAttn output and natural-log LSE candidate requires native package absent from inspected server2 closure. Existing FlashAttentionScore returns max and sum separately; deriving LSE would add prohibited math.', 'MLAPagedFwdOp':'Mixed BF16/FP8 input scales and complete LSE need an original quantized infer API binding; retained exact historical mixed-dtype mapping failure, with no repaired native package evidenced.'}
for s in scope:
 op=s['operator'];cid=s['case_id'];b=s['baseline'];key=(op,cid);r=dict(baseline=b,operator=op,case_id=cid,status='blocked',reason=None,api=None,guard_receipt=None,checker_receipt=None,checked_outputs=0,total_outputs=expected[op]['total_outputs'],registered=False,factory=None,changed_files=[],timings={'queue':None,'device':None,'compile':None})
 f=P/'runs'/cid/'checker.json';g=f.parent/'guard.json'
 if not g.exists():
  for batch in P.glob('*batch.json'):
   if any(item[0]==cid for item in json.load(open(batch))):g=P/'runs'/batch.name/'guard.json';break
 if f.exists() and g.exists():
  c=json.load(open(f));guard=json.load(open(g));v=c.get('verification',{})
  r.update(checker_receipt=str(f.relative_to(P)),guard_receipt=str(g.relative_to(P)),checked_outputs=v.get('checked_outputs',0),total_outputs=v.get('total_outputs',r['total_outputs']),api=c.get('api'),reason=c.get('error'),timings={'queue':None,'device':None,'compile':None,'guard_total':guard.get('elapsed')})
  accepted_teardown = g == P/'runs/dsa_small_batch.json/guard.json' and (P/'TEARDOWN_TIMEOUT_EVIDENCE.json').exists()
  if c.get('status')=='verified' and (guard['returncode']==0 or accepted_teardown):
   r.update(status='completed' if key in reg else 'verified_unregistered',registered=key in reg,factory=('integration700_transformer_attention_misc_dense.gqa' if op=='GQADenseFwdOp' else 'integration700_transformer_attention_misc_decode.factory'))
  else:r.update(status='failed',reason=str(c.get('error') or c.get('traceback') or 'guard nonzero'))
  if accepted_teardown:r.update(reason='Full original numerical checks and cleanup completed before guard124 in Python/TBE process teardown. See TEARDOWN_TIMEOUT_EVIDENCE.json; no guard=0 claim.',guard_exit_code=124,evidence=['TEARDOWN_TIMEOUT_EVIDENCE.json'])
 elif g.exists():
  guard=json.load(open(g))
  r.update(status='failed',reason='Guard ended with code '+str(guard['returncode'])+' before any complete original checker receipt. No numerical success claimed; original input/checker unchanged.',guard_receipt=str(g.relative_to(P)),api='ops-transformer::FlashAttentionScore',evidence=[str((g.parent/'guard.log').relative_to(P))],timings={'queue':None,'device':None,'compile':None,'guard_total':guard.get('elapsed')})
  if (f.parent/'started.json').exists():
   gm=json.load(open(f.parent/'started.json'))['guard'];acq=datetime.datetime.fromisoformat(gm['acquired_at']).timestamp();req=datetime.datetime.fromisoformat(gm['requested_at']).timestamp();r['timings'].update(queue=acq-req,reservation_hold_seconds=guard['end']-acq)
 elif op=='DSADecodeWithKVCacheFwdOp' and cid=='mla-d576-h128-kv128k-bfloat16' and (P/'DSA_TIMEOUT.json').exists():
  r.update(reason='Not run after same original full checker/FlashAttentionScore route hit the 300-second guard limit at 96k. Canonical 128k has identical B/S/H/D/K and larger S_kv; source evidence implies greater dense CPU reference work. Shared execution-budget blocker, not an observed numerical failure. Needs an authorized runtime-budget/environment repair without altering reference/math/input.',api='ops-transformer::FlashAttentionScore',evidence=['DSA_TIMEOUT.json','DSA_REFERENCE_BUDGET_EVIDENCE.json','runs/mla-d576-h128-kv96k-bfloat16/guard.json'])
 elif b=='ops-transformer' and op in block_ops and op!='DSADecodeWithKVCacheFwdOp':r.update(reason=block_ops[op],api='See DECISIONS_SCOPED.json / HISTORY.json',evidence=['DEPLOYMENT.json','SOURCE_PINS.json','HISTORY.json'])
 elif b=='ops-transformer' and op=='GQADenseFwdOp' and 'scaled' in cid:r.update(reason='Retained exact original FlashAttentionScore 500002 compile failure; runtime q_scale and FP8 prefill require a different complete original quantized API. QuantFlashAttn source exists but inspected server2 closure has no compiled extension.',api='QuantFlashAttn',evidence=['HISTORY.json','DEPLOYMENT.json'])
 elif b=='DeepEP-Ascend':r.update(reason='Original EngramBuffer reaches HCCL communicator initialization but fails hcclCommInitRootInfoConfig error 4 / topology JSON parsing. /etc/hccl_rootinfo.json is a read-only mounted directory; no task-private environment-only topology repair established. No unchanged device rerun.',api='EngramBuffer.set_config/write/fetch',evidence=['DEPLOYMENT.json'])
 elif op=='NSAVarlenFwdOp':r.update(reason='Original checker CPU reference moves tensor arguments but leaves workload.g_slc captured on NPU; unchanged reference fails CPU/NPU device mismatch before candidate execution. Shared workload/checker edits prohibited here. Exact proposed checker device-routing patch supplied, not applied; sibling shares the same reference state.',api='FlashAttentionScore',evidence=['runs/prefill-2x4k-float16/checker.json','proposed_checker_nsa_state_device.patch'])
 elif b=='vllm-ascend' and op in ('GDNFwdOp','KDAFwdOp'):r.update(reason='Unchanged checker invokes original FP64 reference on NPU for vllm. GDN representative fails aclnnMatmul 161002: DT_DOUBLE unsupported, before candidate execution. GDN length8192 and KDA references both allocate FP64 state and matrix arithmetic; no dtype reduction or oracle change allowed. Proposed device-only checker patch supplied for parent, not applied.',api='chunk_kda',evidence=['runs/q35-27b-32x8192-float16/checker.json','proposed_checker_cpu_reference.patch'],total_outputs=2)
 elif b=='vllm-ascend' and op=='CumsumFwdOp':r.update(reason='Same original vector scan compile blocker as hidden-state-float16: full scan BT>=4096 materializes BT*BT >=16777216 elements, exceeding Triton maximum tensor numel 1048576. No chunk carry algorithm or compiler/kernel change allowed; siblings not replayed.',api='chunk_local_cumsum_vector',evidence=['runs/hidden-state-float16/checker.json','SOURCE_PINS.json'])
 elif b=='ops-tensor' and op=='SumFwdOp':r.update(reason='Current Sum provider synthesizes reduction as X @ ones via _constant_contraction and GEMM; original ops-tensor source inventory has no standalone complete Sum API for this case. This is not a layout-only binding of an original Sum interface; retained blocker rather than rerunning the old mathematical replacement.',api=None,evidence=['TileOPs/benchmarks/ascend_all_pairs/ops_tensor/reductions.py:_constant_contraction','TileOPs/benchmarks/ascend_all_pairs/ops_tensor/adapters.py:_gemm_engine'])
 else:continue
 if r['status'] in ('completed','verified_unregistered'):
  module=('dense' if op=='GQADenseFwdOp' else 'dsa' if op=='DSADecodeWithKVCacheFwdOp' else 'grouped' if b=='ops-tensor' else 'decode')
  name='integration700_transformer_attention_misc_'+module+'.py'
  r['factory']='benchmarks.ascend_all_pairs.'+('ops_tensor' if b=='ops-tensor' else 'ops_transformer')+'.'+name[:-3]+('.gqa' if module=='dense' else '.factory')
  r['changed_files']=[name]+(['provider/__init__.py'] if r['registered'] else [])
  import hashlib
  r['factory_sha256']=hashlib.sha256((P/name).read_bytes()).hexdigest()
 if f.exists() and g.exists():
  startfile=f.parent/'started.json'
  if startfile.exists():
   stamp=json.load(open(startfile));gm=stamp['guard'];acq=datetime.datetime.fromisoformat(gm['acquired_at']).timestamp();req=datetime.datetime.fromisoformat(gm['requested_at']).timestamp()
   r['timings'].update(queue=acq-req,reservation_hold_seconds=json.load(open(g))['end']-acq,checker_wall_seconds=json.load(open(f)).get('wall_seconds'))
 rows.append(r)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();d={'utc':now,'assigned':len(scope),'reported':len(rows),'rows':rows};tmp=P/'RESULT.json.tmp';tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(P/'RESULT.json')
(P/'HEARTBEAT.json').write_text(json.dumps({'utc':now,'phase':'numerical_batches','current_keys':[], 'last_progress_at':now,'blocked_reason':None,'reported':len(rows),'completed':sum(r['status']=='completed' for r in rows)},indent=2))
print(len(rows),{s:sum(r['status']==s for r in rows) for s in set(r['status'] for r in rows)})
