"""CPU-only evidence projection: stdlib reads, AST parsing, no baseline imports."""
import ast, datetime, hashlib, json, pathlib, shutil
ROOT=pathlib.Path('/root/dyq/workspace'); BASE=ROOT/'reports/baseline-integration-700-20261026'; OUT=BASE/'non_tla_other71/original_ops'
def dump(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def ident(p):
 p=pathlib.Path(p);p=p if p.is_absolute() else ROOT/p
 return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
scope=json.loads((OUT/'SCOPE.json').read_text()); keys={tuple(x['key']) for x in scope}; prior={}
def walk(d):
 if isinstance(d,dict):
  k=d.get('key',d.get('exact_key'))
  if isinstance(k,list) and tuple(k) in keys: prior[tuple(k)]=d
  else:
   for v in d.values():walk(v)
 elif isinstance(d,list):
  for v in d:walk(v)
for p in dict.fromkeys(x['evidence'] for x in scope):walk(json.loads((BASE/p).read_text()))
dump('PRIOR_EXACT_ROWS.json',list(prior.values()))
receipts={
0:'nn_remote_remaining6/case1/checker.json',1:'../baseline-integration-20261026/nn_misc_math/recovery01/runs/SumFwdOp/output/checker.json',2:'nn_remote_remaining6/case5/checker.json',3:'nn_recovery01/ln02_2/checker.json',4:'nn_recovery01/ln03_2/checker.json',8:'surge_dense_linear92/worker03/instance_eval_fp32_readmission/checker.json',9:'nn_remote_remaining6/case3/checker.json'}
projection={}
for i,p in receipts.items():
 path=BASE/p;d=json.loads(path.read_text());projection[str(i)]={**ident(path),'facts':{k:d.get(k) for k in ['op','library','case_id','api','adaptation','status','stage','error','verification','input_seed','case_snapshot','wall_seconds']}}
dump('CHECKER_EVIDENCE.json',projection)
src='TileOPs/benchmarks/ascend_all_pairs/ops_nn/normalization.py'
refs={
0:[('reports/baseline-integration-700-20261026/nn_remote_remaining6/case1/adapter.py',24),('ascend_baselines/ops-math/examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc',425)],
1:[('ascend_baselines/ops-math/math/reduce_sum/op_api/aclnn_reduce_sum.h',22),('reports/baseline-integration-20261026/nn_misc_math/recovery01/sum_adapter.py',10)],
2:[('ascend_baselines/ops-nn/norm/batch_norm_v3/op_host/op_api/aclnn_batch_norm.h',24),(src,31),('TileOPs/workloads/numerics.py',369),('TileOPs/scripts/ascend_pair_check.py',383),('TileOPs/workloads/norm.py',259)],
3:[('ascend_baselines/ops-nn/norm/add_layer_norm/op_host/add_layer_norm_def.cpp',18),('reports/baseline-integration-700-20261026/nn_recovery01/ln02_2/adapter.py',38),(src,84)],
4:[('ascend_baselines/ops-nn/norm/add_layer_norm/op_host/add_layer_norm_def.cpp',18),('reports/baseline-integration-700-20261026/nn_recovery01/ln03_2/adapter.py',38),(src,84)],
5:[('ascend_baselines/ops-nn/matmul/quant_batch_matmul_v3/op_api/aclnn_quant_matmul_v4.h',42),('reports/baseline-integration-700-20261026/threehost_nn_quantmatmul1_r2/adapter.py',5)],
6:[('reports/baseline-integration-700-20261026/nn_recovery01/ln02_2/adapter.py',56)],
7:[('reports/baseline-integration-700-20261026/nn_recovery01/ln02_2/adapter.py',56)],
8:[('ascend_baselines/ops-nn/norm/batch_norm_v3/op_host/op_api/aclnn_batch_norm.h',24),(src,31)],
9:[('reports/baseline-integration-700-20261026/nn_remote_remaining6/case3/adapter.py',32),('ascend_baselines/ops-nn/norm/group_norm_v2/op_host/group_norm_v2_def.cpp',18),(src,23)],
10:[('TileOPs/benchmarks/ascend_all_pairs/ops_tensor/reductions.py',13)]}
notes={
0:('blocked_historical_numeric_failure_and_complete_api_not_established','Historical failure is the per-channel LayerNorm binding, not evidence of a standalone original InstanceNorm API failure. Current scope forbids composing suboperators; do not republish or retry this binding.','Supply a pre-existing complete original ops-math InstanceNorm API and establish contract equivalence; current layer_norm loop is not that API.'),
1:('blocked_prior_device_error_unchanged','Original ReduceSum transposed-axis adapter already exists; historical 507035 invalid GM/stream error is execution failure, not a numeric verdict. No new repair demonstrated.','Prove a source-preserving ABI/layout/deployment correction resolves the exact failed route before scheduling. Existing transpose is already historical, not a new condition.'),
2:('blocked_original_checker_none_coverage','Current reference returns (tensor,None,None); compare_outputs skips empty tensor pairs in coverage count, while outer checker requires checked_outputs == total_outputs. Equivalent output packing cannot repair this.','Requires an explicitly authorized upstream checker/contract decision outside this task; preserve all three outputs and do not drop None or fabricate tensors.'),
3:('blocked_original_numeric_failure_unchanged','Original single AddLayerNorm adapter failed the unchanged checker. Current generic adapter additionally calls LayerNormV4 at normalization.py:87; this composition is inadmissible and is not a repair.','Evidence of a permissible signature/layout/deployment defect correction for original single AddLayerNorm; otherwise retain failure.'),
4:('blocked_original_numeric_failure_unchanged','Original single AddLayerNorm adapter failed the unchanged checker; generic AddLayerNorm+LayerNormV4 cannot be used as replacement.','Evidence of a permissible signature/layout/deployment defect correction for original single AddLayerNorm; otherwise retain failure.'),
5:('prepared_original_adapter_deployment_blocked','Full QuantMatmulV4 signature matches existing adapter: x1,x2,scale,offset,pertokenScale,bias,transposeX1,transposeX2,out. Current known vendor lacks required exports and ARM source header.','Deploy missing untouched quant_batch_matmul_v3 and original CMake dependency closure, build original ARM package, attest API exports and native dispatch; no install3 modified artifacts.'),
6:('blocked_complete_original_grouped_api_not_established','Local ops-nn matmul inventory and inspected remote package do not establish a complete Grouped API. Existing per-group MatMulV3 route is prohibited. Not evidence of global API absence or numeric failure.','Supply pre-existing original ops-nn complete Grouped API/source/build recipe for exact group sizes, transpose and output contract; no MatMul loop or cross-baseline substitution.'),
7:('blocked_complete_original_grouped_api_not_established','Local ops-nn matmul inventory and inspected remote package do not establish a complete Grouped API; no numerical run claimed.','Supply pre-existing original ops-nn complete Grouped API/source/build recipe for exact group sizes, transpose and output contract; no MatMul loop or cross-baseline substitution.'),
8:('blocked_original_numeric_failure_unchanged','Historical FP32 widening readmission already failed original BatchNormV3 eval checker. Widening again is not a changed condition.','Prove permissible original API signature/state/deployment error and its correction; do not widen again, modify tolerance or replace math.'),
9:('blocked_historical_numeric_failure_no_admissible_change','History is per-channel LayerNormV4, not standalone InstanceNorm. Current GroupNormV2 path exists, but changing implementation alone does not authorize replay of failed exact key; no demonstrated binding defect or deployment readiness.','Establish allowed complete original API identity and concrete equivalent interface correction; no per-channel LayerNorm composition. Coordinator must assess before any device run.'),
10:('blocked_complete_original_sum_api_not_established','Current provider computes X @ ones via _constant_contraction and GEMM, not original complete ops-tensor Sum. Source inventory is scoped, not proof of universal absence.','Supply pre-existing complete original ops-tensor Sum API and deployment recipe; do not substitute ops-math or GEMM.')}
shutil.copyfile(BASE/'threehost_nn_quantmatmul1_r2/adapter.py',OUT/'quantmatmul_original_adapter.py')
parsed=[]
for p in {p for rr in refs.values() for p,l in rr if p.endswith('.py')} | {str(OUT/'quantmatmul_original_adapter.py')}:
 path=pathlib.Path(p);path=path if path.is_absolute() else ROOT/path;ast.parse(path.read_text());parsed.append(ident(path))
dump('STATIC_CHECKS.json',{'method':'ast.parse only; no torch, torch_npu, baseline import, CDLL, build or device access','passed':True,'files':parsed,'quant_adapter_byte_identical':(OUT/'quantmatmul_original_adapter.py').read_bytes()==(BASE/'threehost_nn_quantmatmul1_r2/adapter.py').read_bytes()})
dump('SOURCE_INVENTORY.json',{'matmul_directories':sorted(p.name for p in (ROOT/'ascend_baselines/ops-nn/matmul').iterdir() if p.is_dir()),'ops_tensor_sum_name_matches':[str(p.relative_to(ROOT)) for p in (ROOT/'ascend_baselines/ops-tensor').rglob('*') if p.is_file() and ('reduce_sum' in p.name.lower() or p.name.lower()=='sum.py')],'limitation':'Scoped filename inventory plus direct provider reads; not a universal absence proof.'})
rows=[]
for i,s in enumerate(scope):
 key=s['key']; status,note,change=notes[i]; budget=600 if i==5 else 120
 command=['npu-guard','run','--owner','baseline700/original_ops/'+str(i),'--wait','600','--timeout',str(budget),'--','/opt/tileops-all-pairs/bin/allpairs-python','/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_nn11/snapshot/TileOPs/scripts/ascend_pair_check.py','--op',key[1],'--case-id',key[2],'--library',key[0],'--output','<approved-private-remote-output>/checker.json']
 rows.append({'key':key,'status':status,'finding':note,'original_interface_sources':[dict(ident(p),line=l) for p,l in refs[i]],'original_complete_api_established':i not in (0,6,7,9,10),'current_deployment_evidence':'CURRENT_DEPLOYMENT.json','deployment_scope_limit':'Only fixed host and supplied known package paths; empty ACLNN exports do not by themselves disprove graph/JIT support. No device/runtime route executed.','historical_evidence':s['evidence'],'historical_status':s['status'],'historical_failure_reason':projection.get(str(i),{}).get('facts',{}).get('error') or s.get('reason'),'checker_evidence':projection.get(str(i)),'new_condition_really_satisfied':False,'private_adapter':'quantmatmul_original_adapter.py' if i==5 else None,'recipe':'DEPLOYMENT_RECIPE.md','blocking_change_required':change,'minimal_device_validation':{'host':'openlibing-DevEnv-661495','argv':command,'executed':False,'runnable_now':False,'preconditions':'New device authorization + coordinator assignment + documented admissible changed condition + original dependency and private exact-key route deployment. Placeholder output must be allocated by owner. Current provider must not silently route to prohibited generic composite.','budget_s':budget,'wait_s':600,'budget_basis':'Provisional safety cap, not predicted completion; quant original oracle previous 300s timeout stage unknown, 600s is historical handoff cap.'},'device_attempts':[],'runtime_ready':False,'registered':False,'accepted_added':0})
dump('RESULT.json',{'worker':'2e6165a2ed964064','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'CPU original API adaptation/deployment preparation','scope_count':11,'new_device_attempts':0,'new_numerical_successes':0,'publish_candidates':0,'accepted_added':0,'rows':rows,'checks':'STATIC_CHECKS.json','remaining_issues':'All 11 remain blocked; quant adapter prepared but not deployable; no changed condition proved. No memory, provider, ACCEPTED or remote file modified.'})
dump('PUBLISH_CANDIDATES.json',{'candidates':[],'scope_count':11,'reason':'No full original checker success found in exact scope-linked receipts; BatchNorm tensor-only comparison is 1/3 and cannot be promoted.','checker_evidence':'CHECKER_EVIDENCE.json'})
dump('HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'completed','last_substantive_output':'RESULT.json: all 11 keys; quant original adapter; current deployment receipt; no publish candidate','device_attempts':0})
print('Generated 11 exact-key results; AST checks passed; no device execution.')
