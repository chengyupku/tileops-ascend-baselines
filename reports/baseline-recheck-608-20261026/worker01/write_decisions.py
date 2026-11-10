import json, pathlib, datetime, re, collections, yaml
ROOT=pathlib.Path('/root/dyq/workspace')
OUT=ROOT/'reports/baseline-recheck-608-20261026/worker01'
spec=ROOT/'TileOPs/src/tileops/manifest/spec/linear_attention.yaml'
s= yaml.safe_load(spec.read_text())['GDNFwdOp']
rows=json.loads((OUT/'SCOPE.json').read_text())['rows']
lookup={w['label']+'-'+d['T']:(w,d['T']) for w in s['workloads'] for d in w['dtype_cases']}
base=ROOT/'ascend_baselines/ops-transformer/attention/chunk_kda_fwd'
api=base/'op_host/op_api/aclnn_chunk_kda_fwd.cpp'
common=[str(spec)+':87',str(ROOT/'TileOPs/workloads/linear_attention/gdn.py')+':gdn_ref',str(base/'docs/aclnnChunkKdaFwd.md')+':30',str(api)+':309',str(api)+':410',str(api)+':638',str(base/'op_kernel/kda_gate_cumsum_kernel.h')+':DispatchKdaGateCumsum',str(base/'op_kernel/chunk_kda_fwd.cpp')+':95',str(base/'op_kernel/arch35/chunk_kda_fwd_impl.h')+':Run']
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
with (OUT/'decisions.jsonl').open('a') as f:
 for r in rows:
  w,dtype=lookup[r['case_id']]
  assert all(not w.get(k,False) for k in ['use_gate_in_kernel','use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval'])
  assert w['K']==w['V']==128 and 0<w['H']<=w['HV']<=128 and w['HV']%w['H']==0
  lengths=w.get('seq_lens',[w['S']]*w['B'])
  if isinstance(lengths,str):
   m=re.fullmatch(r'repeat\((\d+), (\d+)\)',lengths); assert m
   lengths=[int(m[1])]*int(m[2])
  packed='cu_seqlens' in w.get('some',[])
  assert not packed or (w['B']==1 and sum(lengths)==w['S'] and len(lengths)<=1024)
  line=next(i for i,l in enumerate(spec.read_text().splitlines(),1) if 'label: '+w['label']+'}' in l or 'label: '+w['label']+',' in l)
  contract={k:w[k] for k in ['B','S','H','HV','K','V']}
  contract.update(dtype=dtype,sequence_lengths=lengths,packed=packed,initial_state='float32 supplied' if 'initial_state' in w.get('some',[]) else 'zero',state_v_first=w.get('state_v_first',False),scale='128**-0.5',use_gate_in_kernel=False,use_qk_l2norm_in_kernel=False,use_beta_sigmoid_in_kernel=False,allow_neg_eigval=False,outputs=['o: '+dtype+' [B,S,HV,V]','final_state: float32 [N,HV,V,K]' if w.get('state_v_first') else 'final_state: float32 [N,HV,K,V]'])
  desc=f"B={w['B']},S={w['S']},H/HV={w['H']}/{w['HV']},K=V=128,{dtype},序列长度={lengths}"
  reason=desc+'。原版完整aclnnChunkKdaFwd支持这些维度、QKV dtype及FP32初末状态；将标量log gate无损升FP32并沿K维广播，beta无损升FP32，QKV不转换，传scale=128**-0.5、chunkSize=64、useGateInKernel=false、safeGate=false。通道相同的KDA衰减即GDN标量衰减，不增添数学算子。'
  reason+=('cu_seqlens转为原API的aclIntArray，保留各序列边界。' if packed else '使用BSND原布局。')
  reason+=('传原FP32初态。' if 'initial_state' in w.get('some',[]) else 'initialStateOptional=nullptr采用原API零初态。')
  reason+=('stateVFirst=true由原API完成初末状态转置。' if w.get('state_v_first') else 'stateVFirst=false保持K,V状态布局。')
  reason+='提供FP32 finalStateOut并选择attnOut/finalStateOut两个结果；Aqk/Akk按API要求分配后忽略。仅接口合同静态匹配，不宣称数值通过。'
  result={'key':r['key'],'decision':'要接','reason':reason,'evidence':[str(spec)+':'+str(line)]+common,'contract':contract,'api_or_rejected_routes':{'selected':'aclnnChunkKdaFwdGetWorkspaceSize + aclnnChunkKdaFwd','adaptations':['scalar gate broadcast over K','lossless gate/beta to FP32','cu_seqlens metadata conversion if packed','select original outputs'], 'not_used':['FP16 to BF16 QKV conversion is lossy','FusedGdnDecode unconditionally normalizes Q/K and transforms gates/beta','RecurrentKda short sequence token splitting','ChunkGatedDeltaRule normalized-input restriction']},'decided_at':now()}
  f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush()
result_rows=[json.loads(l) for l in (OUT/'decisions.jsonl').read_text().splitlines()]
assert len(result_rows)==len(rows)==79
assert {tuple(r['key']) for r in result_rows}=={tuple(r['key']) for r in rows}
assert len({tuple(r['key']) for r in result_rows})==79
for r in result_rows:
 for e in r['evidence']: assert pathlib.Path(e.rsplit(':',1)[0]).is_file(),e
(OUT/'DONE.json').write_text(json.dumps({'assigned':79,'completed':79,'decision_counts':dict(collections.Counter(r['decision'] for r in result_rows)),'remaining_keys':[],'blocked_details':[],'checks':['79 unique exact scope keys','all required fields present','all evidence files exist','manifest shapes/dtypes/flags checked per case','no NPU imports/runs; no numerical validation'],'completed_at':now()},ensure_ascii=False,indent=2))
(OUT/'HEARTBEAT.json').write_text(json.dumps({'at':now(),'current_family':'GDNFwdOp complete','completed':79,'blockers':[]},indent=2))
print((OUT/'DONE.json').read_text())
