import json, hashlib, ast
from pathlib import Path
ROOT=Path('/root/dyq/workspace')
OUT=ROOT/'reports/baseline-integration-700-20261026/moe_fullchecker_scratch5'
SRC=ROOT/'.cache/ascend-classic/native-pilot-20261020/ops-math/source'
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
resource=ROOT/'reports/baseline-integration-700-20261026/bf16_moe_resource6/PEAK_MEMORY_PLAN.json'
rows=[r for r in json.loads(resource.read_text())['scope_rows'] if 'masked' not in r['key'][-1]]
keys=[]
for r in rows:
 p=r['manifest_params']; A,B,Y=[r['elements'][k] for k in ('A','B','Y')]; R=p['a'][0][0]; d=4 if p['output'][1]=='float32' else 2
 base=r['phases']['input_preservation']['hbm_explicit_payload_upper_bytes']
 descriptors={}
 for op,param in [('A','a'),('B','b'),('metadata','layout_metadata')]:
  descriptors[op]=dict(shape=p[param][0],dtype=p[param][1],rtol=0,atol=0,equal_nan=True)
 descriptors['checker']=dict(shape=p['output'][0],dtype='float32',rtol=1e-3 if d==4 else 2e-2,atol=1e-2 if d==4 else 1e-1,equal_nan=False)
 keys.append(dict(case_id=r['key'][-1],elements=dict(A=A,B=B,Y=Y,metadata=256),R=R,V=R,output_itemsize=d,
   inherited_live_base_bytes=base,retained_all_expert_FP32_payload_bytes=r['bytes']['retained_native_tensor_payload'],
   input_preservation_success=dict(explicit_python_temp_max_bytes=B+1,formula='max(A,B,256) bool matches + one bool all scalar; comparisons sequential',
       base_plus_explicit_bytes=base+B+1,
       source_conditional_IsClose_output_pair_plus_kernel_bytes=2*B+16777216,
       source_conditional_base_plus_IsClose_payload_and_kernel_bytes=base+2*B+16777216,
       metadata_int32_no_python_cast_on_success=True),
   defined_rows_success=dict(explicit_value_temp_max_bytes=max((8+d)*Y if d==2 else 8*Y,9*Y+1),
       formula='BF16: max(10*C during second selection.float(),9*C+1 during all); FP32:9*C+1; C=V*N. BF16 selections die after float conversion; FP32 float() aliases. Excludes index backend scratch.',
       conservative_all_named_selection_cast_match_sum_bytes=(2*d+9)*Y+1,
       index_prior_allowance_bytes=16*R,index_allowance_is_proven_upper_bound=False,
       source_conditional_compare_value_payload_bytes=10*Y,
       base_plus_explicit_value_bytes=base+max((8+d)*Y if d==2 else 8*Y,9*Y+1)),
   input_B_failure=dict(explicit_large_tensor_temp_peak_bytes=7*B,
       formula='matches B + abs_diff 2*B + abs(expected_flat) 2*B + rel_diff 2*B. Existing actual/expected BF16 inputs already in base. flatten aliases contiguous inputs.',
       base_plus_explicit_peak_bytes=base+7*B,
       exceeds_96_decimal_GB=base+7*B>96_000_000_000,exceeds_96_GiB=base+7*B>96*2**30,
       backend_sum_int64_full_cast_if_materialized_extra_bytes=8*B,
       backend_sum_cast_status='conditional only: torch.sum(matches) returns int64 scalar, but full int64 conversion depends on backend and is NOT proven by Python expression'),
   checker_failure=dict(explicit_value_temp_peak_bytes=21*Y,
       formula='8*C FP32 selected arguments + C matches + 4*C abs_diff + 4*C abs(expected_flat) + 4*C rel_diff, excluding scalars/backend workspaces',
       base_plus_explicit_value_bytes=base+21*Y),
   total_formula='base + max(active checker/input temporaries + their workspace) + sum(all expert V5 workspace) + allocator retention/runtime overhead; do not release any expert buffer before cleanup',
   total_peak_certified=False,original_gate_unchanged=True,probe_descriptors=descriptors))
save('PER_KEY_BOUNDS.json',dict(status='payload_accounting_not_total_HBM_upper_bound',keys=keys,
 shape_corrections={'BOUND_AUDIT.json decode-mlp-up R/V':'32768 text is wrong; manifest and resource numeric accounting use 4096', 'BOUND_AUDIT.json prefill-down R/V':'16384 text is wrong; manifest and resource numeric accounting use 32768'},
 failure_scalar_note='Small scalar results/indices and backend arithmetic/reduction/indexing workspace must be added. No total hard bound or percentage reserve certified.'))
findings={
 'status':'CPU_STATIC_ONLY; no runtime dispatch or SDK/source equivalence proven',
 'source_boundary':'Repository math/opbase source, installed SDK headers and torch_npu wheel are separate evidence. Presence of source or header does not prove installed runtime implementation/version.',
 'python':{'preservation':'TileOPs/workloads/numerics.py:443-452 compares A/B/int32 metadata in full with rtol=atol=0,equal_nan=True; no BF16 .float()',
 'checker':'TileOPs/workloads/moe.py:279-296 selects every defined row and converts each argument to FP32; original tolerances preserved.',
 'success':'.venv/lib/python3.12/site-packages/torch/testing/_comparison.py:1102-1128 matches=torch.isclose(...); if torch.all(matches): return',
 'failure_expressions':{'line325':'total_mismatches = number_of_elements - int(torch.sum(matches))','line340':'actual_flat = actual.flatten(); expected_flat = expected.flatten(); matches_flat = matches.flatten()', 'line344':'if not actual.dtype.is_floating_point and not actual.dtype.is_complex: actual_flat = actual_flat.to(torch.int64); expected_flat = expected_flat.to(torch.int64)', 'line350':'abs_diff = torch.abs(actual_flat - expected_flat)', 'line352':'abs_diff[matches_flat] = 0; max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff, 0)', 'line355':'rel_diff = abs_diff / torch.abs(expected_flat); rel_diff[matches_flat] = 0; torch.max(rel_diff, 0)'},
 'failure_implication':'BF16 B mismatch retains BF16 differences, giving 7*B explicit extra bytes at rel_diff creation. Both mlp-up cases exceed 96 GiB with inherited live base even before V5/backend workspace. Metadata failure promotes two 256-element tensors to int64, not the large BF16 weights.'},
 'isclose':{'api':'math/is_close/op_api/aclnn_isclose.cpp:114-165; isclose.cpp:86-100',
 'dtype':'Matching input dtype required; BF16 supported on RegBase. rtol/atol double converted to float32; no L2 full FP32 input cast. arch35 DAG casts BF16 tiles to float in UB/vector processing, not two global FP32 tensors.',
 'copies':'Contiguous(self/other) invoked; packed ND descriptors do not require materialization semantically, but implementation aliasing not audited. l0 IsClose AllocTensor(bool, full output shape), kernel output then ViewCopy into caller bool output. Thus two full bool outputs logically exist; executor workspace may account for internal one.',
 'workspace':'BroadcastBaseTiling::DoTiling: workspaces[0]=DEFAULT_TILING_WORKSPACE_SIZE=16*1024*1024. L2 total is uniqueExecutor->GetWorkspaceSize(), includes tensor/copy/alignment accounting; cannot replace with 16MiB.',
 'overflow':'B totals 3758096384 and 7516192768 exceed signed int32, latter exceeds uint32. DimensionCollapse stores int64 shapes/products; DoOneDimFormerTiling dimLength, ubOuter, blockFormer are int64. Advance dimLength/ubOuter uint64, dimLen field int64. int32 ubFormer/tileNum are UB tile sizes, not full B. No total-element int32 overflow found on checked packed/equal-shape path; not proof for installed runtime.'},
 'all':{'api':'math/reduce_all/op_api/aclnn_all.cpp:265-325; reduce_all.cpp:27-53',
 'dtype':'Input to torch.all is BOOL, not BF16. Contiguous then Cast BOOL (same dtype); ReduceAll output inferred scalar when all dimensions reduced; Cast to caller BOOL and ViewCopy. ConvertToTensor(dim, INT64) adds tiny axes tensor. Initial AllocTensor(self shape) is followed by INFER_SHAPE; do not assume it allocates a full-size persistent output after inference.',
 'workspace':'opbase ReduceOpTiling::CalcUserWorkSpace(): 16*1024*1024 + (groupR>1 ? vectorCoreNum*CeilAlign(outSize*opDag.maxInputBytes,cacheLineSize) : 0). All-reduce outSize=1 logically; actual tiling and executor overhead need query.',
 'overflow':'Reduce input shapes/axes vector<int64_t>; merged viewShape uint64_t; ReduceTilingData shape/size uint64_t and strides int64_t. int32 shapeSize is rank. No full-numel int32 conversion identified on checked host path; runtime/backend fallback unproven.'},
 'wheel':{'path':'.venv/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so','version':'2.13.0.rc1','git':'462f4e18a8303698aca3997edfcc5dd3b1707968','strings':['aclnnIsCloseGetWorkspaceSize','aclnnIsClose','aclnnAllGetWorkspaceSize','AllKernelNpuOpApi.cpp','../third_party/op-plugin/op_plugin/ops/aclops/AllKernelNpu.cpp','ReduceAllD'],
 'conclusion':'Candidate ACLNN entry points exist; legacy all path also exists. Static strings do not prove dispatch or lack of wrapper conversions.',
 'minimum_runtime_observation':'After parent release under npu-guard: capture current installed dispatcher/DFX op API trace for actual torch.isclose(BF16, zero tolerances,equal_nan=True), torch.isclose(FP32, checker tolerances), torch.all(BOOL), recording resolved DSO path/build-id and actual input/output descriptor dtype/shape/stride and GetWorkspaceSize. A tiny smoke establishes candidate path only; full-shape phase1 query tests large tiling. Confirm no wrapper extra cast/contiguous allocation via profiler or dispatch source before using bounds.'},
 'probe':{'file':'workspace_probe.py','default':'CPU plan only; no native loads/import torch; --device-query explicitly deferred to parent',
 'method':'Real distinct full device allocations for input/other/matches/scalar; packed ND metadata; aclnnIsCloseGetWorkspaceSize then aclnnAllGetWorkspaceSize. No phase2 calls, no workspace allocation, no numerical reference. Values uninitialized because workspace inference must not read tensor data.',
 'run_after_release':'npu-guard run --owner root/moe-fullchecker-query --wait 600 --timeout 120 -- python3 workspace_probe.py --key <case_id> --operand <A|B|metadata|checker> --device-query',
 'coverage':'Query every distinct descriptor/tolerance tuple; B has two shapes, A/checker include correct five-row shapes. All queried on corresponding bool shape. Do not repeat V5 query.',
 'limits':'Not device-executed or native-loaded in worker. ABI based on installed SDK declarations. Querying ACLNN directly alone cannot establish torch dispatch. Process finalization does not reset shared device.'},
 'admission':'No gate changes; root retains scope5/native ownership. Success and failure admission must be distinct; success fit does not promise mismatch diagnostics fit.'}
save('SOURCE_FINDINGS.json',findings)
files=[resource,ROOT/'reports/baseline-integration-700-20261026/moe_large_peak_bounds5/BOUND_AUDIT.json', ROOT/'TileOPs/workloads/numerics.py',ROOT/'TileOPs/workloads/moe.py',ROOT/'TileOPs/src/tileops/manifest/spec/moe.yaml',ROOT/'TileOPs/benchmarks/ascend_math_more.py',ROOT/'.venv/lib/python3.12/site-packages/torch/testing/_comparison.py',ROOT/'.venv/lib/python3.12/site-packages/torch_npu/version.py',ROOT/'.venv/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so']
for op in ['is_close','reduce_all']:
 for sub in ['op_api','op_host/arch35','op_kernel']:
  files.extend(f for f in (SRC/'math'/op/sub).rglob('*') if f.is_file())
base=SRC/'third_party/opbase'
for rel in ['pkg_inc/op_common/atvoss/broadcast/broadcast_tiling.h','pkg_inc/op_common/atvoss/broadcast/broadcast_tiling_base.h','pkg_inc/op_common/atvoss/broadcast/broadcast_base_struct.h','src/op_common/atvoss/broadcast/broadcast_tiling.cpp','src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp','pkg_inc/op_common/atvoss/reduce/reduce_tiling.h','pkg_inc/op_common/atvoss/reduce/reduce_tiling_data.h','src/op_common/atvoss/reduce/reduce_tiling.cpp']: files.append(base/rel)
sdk=Path('/usr/local/Ascend/cann-9.2.0-beta.2/include')
files.extend(sdk/f for f in ['aclnnop/aclnn_isclose.h','aclnnop/aclnn_all.h','aclnn/acl_meta.h','acl/acl_base_rt.h'])
files.extend(OUT/f for f in ['SOURCE_FINDINGS.json','PER_KEY_BOUNDS.json','workspace_probe.py','build_reports.py'])
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
save('FILEHASHES.json',{'algorithm':'sha256','files':[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))]})
ast.parse((OUT/'workspace_probe.py').read_text())
save('DELIVERY.json',dict(status='complete_CPU_static_deliverable_runtime_measurement_pending',changed_files=['SOURCE_FINDINGS.json','PER_KEY_BOUNDS.json','FILEHASHES.json','DELIVERY.json','workspace_probe.py','build_reports.py'],checks=['JSON serialization','probe ast.parse (no compilation)','manifest/source shape cross-check','integer byte arithmetic','source/wheel SHA256','all 20 default CPU probe plans; no torch/native load branch entered','executor destructor order verified against parent FEEDBACK.txt'],cpu_only=True,NPU_used=False,torch_imported=False,native_libraries_loaded=False,compiled=False,reference_executed=False,shared_state_changed=False,agents_spawned=False,accepted_added=0,remaining=['parent current-runtime dispatch observation','parent reserved workspace phase1 queries','V5 workspace handled independently by parent','allocator/backend indexing and failure reductions remain unbounded'],headline='Two mlp-up B-mismatch diagnostic explicit live payloads already exceed 96GiB; source IsClose adds full bool intermediate and 16MiB kernel workspace, executor total unmeasured.'))
for r in keys: print(r['case_id'],r['input_preservation_success']['base_plus_explicit_bytes'],r['input_B_failure']['base_plus_explicit_peak_bytes'])
