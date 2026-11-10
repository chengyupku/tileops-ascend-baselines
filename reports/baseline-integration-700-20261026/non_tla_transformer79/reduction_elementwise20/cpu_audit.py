"""Read-only CPU audit; writes only this report directory. No device imports."""
import ast, collections, datetime, hashlib, json, pathlib, re, subprocess
ROOT=pathlib.Path('/root/dyq/workspace'); OUT=pathlib.Path(__file__).parent; BASE=ROOT/'reports/baseline-integration-700-20261026'
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ref(path,start,end):
 p=ROOT/path; lines=p.read_text().splitlines(); return dict(path=str(p),line=start,end_line=end,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),text='\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1,min(end,len(lines)))))
scope=json.loads((OUT/'SCOPE.json').read_text()); keys={tuple(r['key']) for r in scope}
dump('HEARTBEAT.json',dict(utc=stamp(),phase='write_exact_contract_results',npu_used=False,next='CPU validation and final handoff'))
prior=json.loads((BASE/'surge_dense_linear92/RESULT.json').read_text())['rows']; prior={tuple(r['key']):r for r in prior if tuple(r['key']) in keys}
evidence=[]
def add(p,s,e):
 evidence.append(ref(p,s,e)); return len(evidence)-1
P='TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'; S='ascend_baselines/ops-transformer/'; M='TileOPs/src/tileops/manifest/spec/'
family={
'AmaxFwdOp':[add(P+'selection_values.py',18,40),add(S+'moe/moe_gating_top_k_softmax_v2/README.md',14,38),add(M+'reduction.yaml',137,158)],
'ClampScalarFwdOp':[add(P+'selection_values.py',58,65),add(P+'selection_values.py',83,90),add(P+'pointwise_extrema.py',24,69),add(S+'experimental/moe/moegategeluclamp/README.md',11,40),add(M+'elementwise_unary_activation.yaml',323,347)],
'LerpTensorFwdOp':[add(P+'composed_activations.py',19,26),add(P+'composed_activations.py',44,51),add(M+'elementwise_multi_input.yaml',31,52)],
'LogSumExpFwdOp':[add(P+'log_reduction.py',14,37),add(S+'attention/attention_update/op_host/op_api/aclnn_attention_update.h',30,49),add(S+'attention/attention_update/op_host/attention_update_tiling.cpp',190,212),add(M+'reduction.yaml',61,81)],
'InstanceNormFwdOp':[add(P+'adapters.py',541,568),add(S+'posembedding/norm_rope_concat/op_host/op_api/aclnn_norm_rope_concat.h',76,136),add(M+'norm.yaml',306,337)]}
# Inventory public entry files, not kernel internal names: scoped absence evidence only.
files=sorted(p for p in (ROOT/S).rglob('*') if p.is_file() and (('op_api' in p.parts and p.suffix=='.h') or p.name.endswith('_def.cpp')))
terms=re.compile(r'lerp|interpolat|instance.?norm|reduce.?max|logsumexp|clamp',re.I)
hits=[]
for p in files:
 for n,line in enumerate(p.read_text(errors='replace').splitlines(),1):
  if terms.search(line): hits.append(dict(path=str(p),line=n,text=line))
dump('PUBLIC_API_INVENTORY.json',dict(utc=stamp(),source_commit=subprocess.check_output(['git','-C',str(ROOT/S),'rev-parse','HEAD'],text=True).strip(),files=[str(p) for p in files],query=terms.pattern,matches=hits,limitation='Scoped public entry inventory, not a universal impossibility claim; fused clamp hits do not provide standalone clamp.'))
# Walk only receipt paths explicitly linked from SCOPE evidence.
receipt_files={BASE/'surge_dense_linear92/RESULT.json',BASE/'threehost_dense48_r2/RESULT.json'}
for r in prior.values():
 for f in r.get('all_current_receipts',[]): receipt_files.add(BASE/'surge_dense_linear92'/f)
projections=[]
def walk(x):
 if isinstance(x,dict):
  key=x.get('key')
  if key is None and all(k in x for k in ['baseline','operator','case_id']): key=[x['baseline'],x['operator'],x['case_id']]
  if key and isinstance(key,list) and tuple(key) in keys: yield x
  for v in x.values(): yield from walk(v)
 elif isinstance(x,list):
  for v in x: yield from walk(v)
for p in sorted(receipt_files):
 if not p.exists(): projections.append(dict(path=str(p),missing=True));continue
 txt=p.read_text(); xs=[json.loads(l) for l in txt.splitlines() if l.strip()] if p.suffix=='.jsonl' else [json.loads(txt)]
 for x in xs:
  for r in walk(x): projections.append(dict(path=str(p),key=r.get('key',[r.get('baseline'),r.get('operator'),r.get('case_id')]),status=r.get('status'),checker=r.get('checker',r.get('checker_receipt')),native=r.get('native'),capture=r.get('capture'),registration_pins=r.get('registration_pins')))
dump('PRIOR_SUCCESS_REVIEW.json',dict(utc=stamp(),receipts=projections,complete_success_found=False,note='No full success receipt in scoped evidence chain; no checker/native/capture/registration_pins success quartet.'))
failure_path=BASE/'transformer_dense/norm_check_graph_optional/InstanceNormFwdOp.json'; fail=json.loads(failure_path.read_text());
dump('HISTORICAL_NUMERICAL_FAILURE.json',dict(path=str(failure_path),sha256=hashlib.sha256(failure_path.read_bytes()).hexdigest(),receipt={k:fail[k] for k in ['case_id','input_seed','case_snapshot','status','error','api','runtime_host','provider_source_changed','oracle_source_changed','source_sha256','adaptation']}))
for p in ['selection_values.py','pointwise_extrema.py','composed_activations.py','log_reduction.py','adapters.py','native.py']:
 ast.parse((ROOT/P/p).read_text())
dump('SOURCE_EVIDENCE.json',dict(utc=stamp(),families=family,evidence=evidence,static_parse='6 provider files parse; no modules imported'))
# Fresh SSH filesystem-only probe, preserve complete output. No dlopen or framework imports.
remote=r'''import pathlib,json,hashlib,shutil,platform
base=pathlib.Path('/opt/tileops-all-pairs/workspace'); c=base/'.cache/all-pairs'; state=c/'resume22_server2_transformer_dense_state01'; src=state/'private_source/gate_20261022T135804/ascend_baselines/ops-transformer'
paths=[c/'ops_transformer/install/vendors/allpairs_transformer',c/'ops_transformer/install/vendors/selectionpairs_transformer',c/'ops_transformer/libtransformer_infer_shim.so',base/'.cache/ascend-classic/native-pilot-20261020/ops-transformer/install/vendors/classicpilot_transformer',state/'native_build/selection/source/build/libcust_opapi.so',state/'native_build/selection/source/build/version.ops-transformer.info',pathlib.Path('/opt/tileops-ci/runtime/bin/python')]
rows=[]
for p in paths:
 r={'path':str(p),'exists':p.exists(),'resolved':str(p.resolve())}
 if p.is_file() and (p.suffix=='.so' or p.suffix=='.info'): r['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.is_file() and p.suffix=='.info': r['text']=p.read_text(errors='replace')
 rows.append(r)
source=[]
for rel in ['attention/attention_update/op_host/attention_update_tiling.cpp','posembedding/norm_rope_concat/op_host/op_api/aclnn_norm_rope_concat.h','moe/moe_gating_top_k_softmax_v2/README.md']:
 p=src/rel;source.append({'path':str(p),'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None})
print(json.dumps({'hostname':platform.node(),'architecture':platform.machine(),'commands':{n:shutil.which(n) for n in ['python3','bisheng','npu-guard','readelf','nm']},'paths':rows,'original_source_files':source},indent=2))
'''
proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','openlibing-DevEnv-055409','python3 -I -S -'],input=remote,text=True,capture_output=True,timeout=45)
dump('REMOTE_DEPLOYMENT.json',dict(utc=stamp(),host='openlibing-DevEnv-055409',command='python3 -I -S - (stdlib filesystem metadata only)',probe_source=remote,exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,limitation='Expected provider paths absent; alternate selection build observed, not proven complete deployable runtime. No device/runtime ABI load performed.'))
contracts={
 'AmaxFwdOp':dict(shape=[2048,11008],dtype='float32',dim=-1,keepdim=False,output=[2048]),
}
reasons={
'AmaxFwdOp':'FP32 current provider selects raw input values by indices then recursively reduces 2048-wide chunks; original TopKSoftmaxV2 returns normalized scores, not a full raw max reduction. No complete matching public entry located.',
'ClampScalarFwdOp':'Two bounds require separate extrema computations. FP32 uses TopKSoftmaxV2/gather; FP16/BF16 uses FlashAttentionScore maxima plus GroupedMatmulAdd negation. Fused MoeGateGeluClamp requires GeGLU, BF16, output width halving, and cannot disable activation. No standalone exact clamp public entry located.',
'LerpTensorFwdOp':'Provider computes input-dependent 1-weight through weighted_rows then interpolation with another weighted call. MoeFinalizeRoutingV2 accepts supplied scales but does not compute tensor lerp coefficients; this is mathematical composition, not signature adaptation.',
'LogSumExpFwdOp':'AttentionUpdate accepts lists of FP32 log states and local values, updateType=1 returns LSE; original host limits sp to [1,16]. Exact width exceeds16; current provider requires a reduction tree. No complete exact reduction API located.',
'InstanceNormFwdOp':'NormRopeConcat has input-statistics LayerNorm modes and optional per-row mean/rstd outputs, but no supplied running-mean/variance input or momentum update API.'}
rows=[]
for item in scope:
 key=item['key']; _,op,case=key; old=prior[tuple(key)]; dtype=case.rsplit('-',1)[1]; status='complete_api_unsupported'; contract={}; historical=None
 if op=='AmaxFwdOp': contract=contracts[op]
 if op in ('ClampScalarFwdOp','LerpTensorFwdOp'):
  size=4096 if case.startswith('16M') else 16384; contract={'shape':[size,size],'dtype':dtype,'output_shape':[size,size]}
  contract.update({'min':-0.5,'max':0.5} if op=='ClampScalarFwdOp' else {'input_shape':[size,size],'end_shape':[size,size],'weight_shape':[size,size],'broadcast':'identical full tensor shapes'})
 if op=='LogSumExpFwdOp': contract={k:old[k] for k in ['shape','reduction_dims','reduction_width']};contract.update(dtype=dtype,keepdim=False)
 if op=='InstanceNormFwdOp':
  contract={'shape':[4,256,28,28] if case.startswith('wider') else [8,128,32,32],'dtype':dtype,'use_input_stats':not case.startswith('image-eval'),'momentum':0.1,'eps':1e-5,'affine':not case.startswith('image-track'),'running_state':not case.startswith('wider'),'checked_outputs':3 if case.startswith('image-track') else 1}
  if case.startswith('wider'): status='original_numerical_failure';historical={'receipt':str(failure_path),'seed':fail['input_seed'],'error':fail['error'],'causal_limit':'Observed historical complete mapped original route fails; does not prove universal kernel defect. No verified adapter-only correction found.'}
 reason=fail['error'] if historical else reasons[op]
 row=dict(key=key,status=status,contract=contract,complete_original_api_exists=(True if historical else False),api_existence_scope='Current inspected original source/public entry inventory and mapped provider, not universal absence',reason=reason,source_evidence_indices=family[op],source_evidence_path=str(OUT/'SOURCE_EVIDENCE.json'),historical_numerical_failure=historical,prior_evidence=str(BASE/item['evidence']),current_deployment_evidence=str(OUT/'REMOTE_DEPLOYMENT.json'),deployment_ready=False,changed_condition_proven=False,changed_condition_required='A demonstrable existing complete original API mapping satisfying this exact contract, with original-source identity and no composition.' if not historical else 'A concrete semantic-equivalent signature/layout defect correction explaining the observed mismatch; current investigation found none.',next_legal_action='acceptance_root retain unsupported status; reopen only upon a concrete new original complete API candidate; no NPU retry.' if not historical else 'acceptance_root retain original numerical failure and seed2442836156; no unchanged retry. Review a concrete adapter-only correction if later evidenced.',checker_command=None,guard_budget_seconds=0,guard_budget_reason='No legal device-ready candidate; no run requested.',new_device_execution=False,verified_pending_publication=False,accepted=False)
 rows.append(row)
 dump('RESULT.json',dict(utc=stamp(),worker='a1a85cac03b04976',scope_count=20,rows=rows,completed_rows=len(rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),verified_pending_publication=0,cpu_ready_for_guarded_check=0,accepted_added=0,npu_used=False,checker_or_benchmark_executed=False,publication_owner='acceptance_root'))
dump('CPU_CHECKS.json',dict(utc=stamp(),unique_keys=len({tuple(r['key']) for r in rows}),matches_scope={tuple(r['key']) for r in rows}==keys,source_evidence_exists=all(pathlib.Path(e['path']).exists() for e in evidence),provider_ast_parse_count=6,npu_imports=False,remote_probe_returncode=proc.returncode))
dump('HEARTBEAT.json',dict(utc=stamp(),phase='completed',last_substantive_output='RESULT.json:19 complete_api_unsupported,1 original_numerical_failure',npu_used=False,next='acceptance_root review; no legal unchanged retry',blocker='No complete original API newly identified; expected remote provider paths missing'))
print(json.dumps({'rows':len(rows),'counts':dict(collections.Counter(r['status'] for r in rows)),'remote_rc':proc.returncode}))
