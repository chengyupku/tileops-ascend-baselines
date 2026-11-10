import json,pathlib,datetime,yaml,hashlib
p=pathlib.Path(__file__).resolve().parent;root=p.parents[2];s=json.loads((p/'SCOPE.json').read_text());spec={}
for f in (root/'TileOPs/src/tileops/manifest/spec').glob('*.yaml'):
 try:spec.update(yaml.safe_load(f.read_text()) or {})
 except Exception:pass
apis={'AmaxFwdOp':'MoeGatingTopKSoftmaxV2','ClampScalarFwdOp':'MoeGatingTopKSoftmaxV2','FusedTopKFwdOp':'aclnnMoeGatingTopKV2','GeluTanhAndMulFwdOp':'torch.ops.ascend_ops.bias_gate_gelu','InstanceNormFwdOp':'NormRopeConcat','LerpTensorFwdOp':'MoeFinalizeRoutingV2 (legacy composed adapter; not validated)','LogSumExpFwdOp':'AttentionUpdate','LongRoPEFwdOp':'aclnnRotaryPositionEmbedding','YaRNFwdOp':'aclnnRotaryPositionEmbedding','MoEPrePermuteFwdOp':'MoeInitRoutingV2','MoEPostPermuteFwdOp':'MoeFinalizeRoutingV2','PagedKVCacheWriteFwdOp':'ScatterPaKvCache','HadamardTransformFwdOp':'GroupedMatmul (legacy decomposed adapter; not validated)','FusedMoEExpertsFwdOp':'complete fused experts API not deployed'}
rows=[]
for x in s:
 op=x['operator'];api=apis.get(op,'aclnnGroupedMatmulV5' if 'Gemm' in op or 'Bmm' in op else 'GroupedMatmulAdd (legacy convolution adapter; not validated)')
 reason=f'{api}: assigned aarch64 server1 has no matching original ops-transformer host/device/JIT deployment in inspected existing package roots. Present original transformer transfer exports only ChunkKdaFwd/RecurrentKda (outside scope); FLA package exports its own delta APIs. Existing root provider only registers BMM. A complete new vendor build/framework bootstrap is not a permitted minimal adapter repair.'
 evidence=['HOST_INVENTORY.json','SYMBOLS.json']
 if 'Gemm' in op or 'Bmm' in op:
  reason+=' Prior pristine GroupedMatmul compilation also fails at Blaze asc::te versus bundled AscendC::Te; no compatible repaired original package found, so unchanged build was not repeated.';evidence+=['prior_dense_DELIVERY.json','prior_dense_BLOCKER_EVIDENCE.json']
 if op=='FusedTopKFwdOp':
  reason='Existing vLLM custom_transformer exports V1 MoeGatingTopK, but its deployed kernel hashes differ from current original ops-transformer; no original V2 package, nor provenance establishing equivalence. Substituting vLLM implementation for ops-transformer is not accepted.';evidence+=['TOPK_CANDIDATE.txt','TOPK_ORIGINAL_HASHES.txt']
 if op=='GeluTanhAndMulFwdOp':
  reason='Private original-source build attempted; final build/checker status recorded separately in BUILD.json. Not yet numerically verified or registered.';evidence+=['BUILD.json']
 outs=spec.get(op,{}).get('signature',{}).get('outputs',{})
 rows.append({**{k:x[k] for k in ['baseline','operator','case_id']},'status':'blocked','reason':reason,'api':api,'guard_receipt':None,'checker_receipt':None,'checked_outputs':0,'total_outputs':len(outs)+sum(bool(v.get('mutated')) for v in spec.get(op,{}).get('signature',{}).get('inputs',{}).values()),'registered':False,'factory':None,'changed_files':[],'timings':{'queue_s':0,'device_s':0,'compile_s':None},'evidence':evidence,'classification_preserved':True})
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':'server1','scope_count':len(s),'completed':0,'rows':rows};tmp=p/'RESULT.json.tmp';tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(p/'RESULT.json')
