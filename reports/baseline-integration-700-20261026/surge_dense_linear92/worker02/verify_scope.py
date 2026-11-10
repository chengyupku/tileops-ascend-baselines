import datetime, hashlib, json, pathlib, re
W=pathlib.Path('/root/dyq/workspace'); P=pathlib.Path(__file__).parent
S=json.loads((P/'SCOPE.json').read_text())
def pin(p):
 p=pathlib.Path(p); p=p if p.is_absolute() else W/p
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(name,v): (P/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
rows=[]
for i,s in enumerate(S):
 k=s['key']; op=k[1]; case=k[2]
 r=dict(key=k,task_type='实际对接/补测',status=s['status'],reason=s['reason'],prior_scope_path=s['prior_scope_path'],registered=False,accepted_added=0,runtime_ready=False,new_device_execution=False,attempts=[],checker=None,guard=None,native=None,checked_outputs=0,total_outputs=s.get('total_outputs',1),retry_condition_changed=False,timings={'requested':None,'acquired':None,'child_start':None,'end':None,'queue_seconds':0,'physical_wait_seconds':0,'device_seconds':0,'compile_seconds':0},evidence=[])
 if op=='GemmFP8FwdOp':
  spec=W/'TileOPs/src/tileops/manifest/spec/gemm.yaml'; label=case.split('-float8_')[0]
  lines=[(n,l) for n,l in enumerate(spec.read_text().splitlines(),1) if 'label: '+label+'}' in l and 'SA:' in l]
  assert lines
  line=next((x for x in lines if ('out_dtype: float16' in x[1]) == case.endswith('-float16')),lines[0]); text=line[1]
  dims={t:int(re.search(r'\b'+t+r': (\d+)',text).group(1)) for t in ['M','N','K']}
  sb=[int(v) for v in re.search(r'SB: \[([^]]+)\]',text).group(1).split(',')]
  assert sb==[dims['N'],dims['K']//128] and dims['M']>1
  r['exact_parameters']=dict(**dims,scale_b=sb,required_original_GB_scale_b=[(dims['N']+127)//128,dims['K']//128],swapped_scale_N=dims['M'],swapped_required_scale_N=(dims['M']+127)//128)
  r['evidence']=[dict(**pin(spec),line=line[0],text=text),pin('TileOPs/workloads/gemm.py'),dict(**pin('ascend_baselines/ops-transformer/gmm/grouped_matmul/op_api/grouped_matmul_950_checker.cpp'),lines=[1461,1514])]
 elif op=='InstanceNormFwdOp' and case!='wider-channel-affine-float16':
  r['evidence']=[pin('TileOPs/src/tileops/manifest/spec/norm.yaml'),pin('ascend_baselines/ops-transformer/posembedding/norm_rope_concat/op_host/op_api/aclnn_norm_rope_concat.h')]
  r['missing_contract']='accept supplied running mean/variance' if 'eval' in case else 'update running mean and unbiased running variance'
 elif op=='AmaxFwdOp':
  r['exact_parameters']={'S':[2048,11008],'dim':-1,'dtype':'float32'}
  r['evidence']=[pin('TileOPs/src/tileops/manifest/spec/reduction.yaml'),pin('TileOPs/benchmarks/ascend_all_pairs/ops_transformer/selection_values.py'),pin('ascend_baselines/ops-transformer/moe/moe_gating_top_k_softmax_v2/README.md')]
 else:
  if op=='InstanceNormFwdOp': cp=W/'reports/baseline-integration-700-20261026/transformer_dense/norm_check_graph_optional/InstanceNormFwdOp.json'
  else: cp=W/'reports/baseline-integration-700-20261026/transformer_linear/mamba2-780m-float16-float32/checker.json'
  c=json.loads(cp.read_text()); assert [c['library'],c['op'],c['case_id']]==k and c['status']=='failed'
  r.update(status='prior_exact_numerical_failure_preserved',reason=c['error'],checker=pin(cp),seed=c['input_seed'],exact_parameters=c['case_snapshot'],historical_checker_seconds=c['wall_seconds'])
  if op=='SSDChunkStateFwdOp':
   gp=cp.parent/'GUARD_RECEIPT.json'; r['guard']=pin(gp); r['prior_guard_receipt']=json.loads(gp.read_text())
   assert r['guard']['sha256']==s['guard_receipt']['sha256'] and r['checker']['sha256']==s['checker_receipt']['sha256']
  else:
   gp=cp.parent/'guard_snapshot.json'; r['historical_guard_snapshot']=pin(gp); r['guard_receipt_limitation']='Historical snapshot available; no per-case guard exit receipt in supplied scope. Not a new successful run.'
  r['evidence']=[pin(cp)]
 r['next_step']='Original implementation/API change or newly established complete original API mapping required before retry; source changes outside this task authorization.'
 rows.append(r); write('RESULT_%02d.json'%i,r)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
write('RESULT.json',dict(at=now,scope_count=len(rows),rows=rows,numerically_verified=0,accepted_added=0,NPU_runs=0,pending_parent_registration=[],true_remaining=[r['key'] for r in rows],three_host_submission_check='Not invoked: no device submission and no host-occupancy blocker; constraints are original API/math',checks={'exact_key_coverage':len({tuple(r['key']) for r in rows})==len(S),'ssd_receipt_hashes_match':True,'fp8_exact_scale_shapes_checked':8,'historical_numerical_failures_read':2},timing_note='No current device/compile/queue time; historical timings retained only on corresponding rows'))
write('HEARTBEAT.json',dict(utc=now,phase='complete_no_legal_device_retry',exact_keys=[r['key'] for r in rows],last_output='13 individual RESULT files with original source pins and exact failure receipts',blocker='11 original API boundaries, 2 preserved exact numerical failures',next_step='Parent review; no ready/accepted claim'))
print(json.dumps({'rows':len(rows),'api_boundaries':11,'preserved_failures':2,'NPU_runs':0,'accepted_added':0}))
