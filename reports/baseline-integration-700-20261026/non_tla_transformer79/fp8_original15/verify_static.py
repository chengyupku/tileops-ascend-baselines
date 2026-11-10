"""CPU-only contract audit. Uses stdlib; never imports baseline/Torch/device packages."""
import ast, datetime, hashlib, json, pathlib, re
P = pathlib.Path(__file__).resolve().parent
W = P.parents[3]
B = W/'reports/baseline-integration-700-20261026'
def read(p): return (W/p).read_text()
def evidence(p, lines):
    return dict(path=str(W/p), lines=lines, sha256=hashlib.sha256((W/p).read_bytes()).hexdigest())
def write(name, data): (P/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
manifest='TileOPs/src/tileops/manifest/spec/gemm.yaml'
workload='TileOPs/workloads/gemm.py'
api='ascend_baselines/ops-transformer/gmm/grouped_matmul/op_api/'
docs='ascend_baselines/ops-transformer/gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md'
provider='TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'
scope=json.loads((P/'SCOPE.json').read_text()); keys={tuple(r['key']) for r in scope}
assert len(keys)==15
histories=['surge_dense_linear92/RESULT.json','surge_dense_linear92/worker02/RESULT.json','surge_dense_linear92/worker06/RESULT.json','threehost_dense48_r2/RESULT.json']
historical={k:[] for k in keys}
for h in histories:
    data=json.loads((B/h).read_text())
    for r in data['rows']:
        key=tuple(r.get('key',[]))
        if key in keys:
            historical[key].append(dict(path=str(B/h), status=r.get('status'), checker=r.get('checker'), native=r.get('native'), checked_outputs=r.get('checked_outputs'), evidence=r.get('evidence')))
# Execute only extracted shape metadata method, no source module import.
tree=ast.parse(read(workload)); cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GemmFP8Workload')
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_scale_shapes')
ns={}; exec(compile(ast.Module(body=[fn],type_ignores=[]),workload,'exec'),ns)
common=[evidence(workload,[110,179]),evidence(api+'aclnn_grouped_matmul_v5.h',[66,85]),evidence(api+'grouped_matmul_950_checker.cpp',[1408,1514]),evidence(docs,[448,500]),evidence(provider+'adapters.py',[942,976]),evidence(provider+'__init__.py',[218,219]),evidence(provider+'_integration700_fp8_m1/__init__.py',[14,16]),evidence(provider+'fp8_column_scales.py',[33,48]),evidence(provider+'native.py',[12,20])]
common.append(evidence('TileOPs/scripts/ascend_pair_check.py',[309,391]))
common.append(evidence('TileOPs/workloads/numerics.py',[158,198]))
rows=[]
for s in scope:
    key=s['key']; suffix='-float8_e4m3fn-'+key[2].rsplit('-',1)[1]; label=key[2][:-len(suffix)]
    matches=[(i,l) for i,l in enumerate(read(manifest).splitlines(),1) if i>=86 and 'label: '+label+'}' in l and 'float8_e4m3fn' in l and ('out_dtype: float16' in l)==key[2].endswith('-float16')]
    assert len(matches)==1,(key,matches)
    line,text=matches[0]; dims={n:int(re.search(r'\b'+n+r': (\d+)',text)[1]) for n in ('M','N','K')}
    m,n,k=(dims[x] for x in ('M','N','K')); sa=[m,k//128]; sb=[n,k//128]
    obj=type('Metadata',(),dict(m=m,n=n,k=k,scale_mode='block128'))()
    assert ns['_scale_shapes'](obj)==(tuple(sa),tuple(sb))
    assert m>1 and n>1 and k%128==0 and n!=(n+127)//128 and m!=(m+127)//128
    hist=historical[tuple(key)]
    assert all(not h['checker'] and not h['native'] for h in hist),hist
    rows.append(dict(key=key,status='complete_api_unsupported',contract=dict(**dims,A=[m,k],B=[n,k],SA=sa,SB=sb,scale_dtype='float32',input_dtype='float8_e4m3fn',out_dtype=key[2].rsplit('-',1)[1],output=[m,n],bias=False,reference='(A.float()*expand(SA)) @ (B.float()*expand(SB)).T; cast to out_dtype',checker='ascend_pair_check.py:377-391: original verify with case.verification, preserve_inputs=True, checked_outputs==total_outputs, FP8 edge check; unchanged seed/tolerance'),source=evidence(manifest,[line,line]),public_api=dict(exists=True,name='aclnnGroupedMatmulV5GetWorkspaceSize + aclnnGroupedMatmulV5',signature_source=common[1],exact_contract_supported=False,direct_native_scale=[1,k//128,(n+127)//128],actual_transposed_scale=[1,k//128,n],swapped_native_scale=[1,k//128,(m+127)//128],swapped_actual_scale=[1,k//128,m]),provider_difference='Dispatch enters fp8_block_gemm and rejects independent row scales at adapters.py:958-959; this agrees with original API. Registered M1 wrapper exact key is outside scope. Column-loop staging is forbidden composition, not eligible.',historical=hist,original_numerical_failure=False,numerical_failure_note='No scoped full checker/native execution evidence in the cited result chain; unsupported metadata is not a numerical failure.',verified_pending_publication=False,checker=None,native=None,capture=None,registration_pins=None,changed_condition_proven=False,changed_condition_required='An unchanged original public complete API supporting E4M3 operands with independent FP32 1x128 scales on BOTH sides, or a proof of lossless complete-API metadata binding for these exact dimensions. Deployment repair alone is insufficient.',next_legal_action='acceptance_root records unsupported without acceptance or device retry; reopen only on the explicit API/mapping evidence above. Do not execute fp8_column_scales.',deployment_evidence=['DEPLOYMENT_FILE_CHECK.json','DEPLOYMENT_PROVIDER_CHECK.json','DEPLOYMENT_DISCOVERY.json'],deployment_status='Expected private-provider and default GraphCall vendor paths absent on authorized host; no runtime ABI/export identity established.',runtime_ready=False,guard_budget_seconds=0,checker_command=None,checker_command_reason='No eligible candidate; supplying a runnable device command would be an unchanged unsupported retry.',accepted=False))
write('RESULT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker='14d734782da34bd3',task='original complete FP8 API / deployment CPU audit',scope_count=15,rows=rows,common_source_evidence=common,summary={'complete_api_unsupported':15,'cpu_ready_for_guarded_check':0,'original_numerical_failure':0,'verified_pending_publication':0,'accepted_added':0},api_alternatives_review={'G-B':'FP32 scales but fixed gsM=1,gsN=gsK=128; both direct and swapped binding fail N scale granularity.','MX_G-G':'gsK=32 and FLOAT8_E8M0 scales; arbitrary FP32 random scales cannot be cast losslessly.','T_T_T_C_K_T_K_C':'Tensor/token/channel factors do not represent independent K-block scales.','V3_V4':'Legacy entries in aclnn_grouped_matmul.cpp:3491-3602 feed shared grouped matmul machinery; no independent FP32 G-G capability exposed.','WeightNz':'Weight layout API does not supply a general independent FP32 G-G FP8 mode; no evidence permitting replacement of exact scales.','column_loop':'Existing unregistered source composes one product per column; explicitly forbidden in this task.'},new_device_runs=0,device_imports=0,checker_or_benchmark_runs=0,thin_adapter=None,thin_adapter_reason='No semantically equivalent original complete API candidate established.',deployment_recipe='Do not deploy for these unsupported keys. If new complete-API evidence appears, coordinator must assign host; CPU-stage only matching original aarch64 vendor package and source identities into private task directory, inspect exports/dependencies and package metadata, then acceptance_root prepares original checker under host guard. Existing M1 deployment is server1-specific and outside scope.',limitations=['Remote read-only file checks do not establish driver/runtime readiness or loaded binary identity.','Success search restricted to SCOPE evidence chain; no claim of globally searching unrelated records.','No new numerical conclusions from CPU inspection.']))
write('CPU_STATIC_CHECKS.json',dict(passed=True,scope_rows=15,unique_keys=15,manifest_exact_matches=15,extracted_workload_scale_shape_checks=15,direct_and_swapped_mismatch_checks=30,torch_imported=False,baseline_imported=False,historical_receipts_checked=histories,successful_scoped_checker_found=0))
write('HEARTBEAT.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='complete',scope_count=15,last_substantive_output='RESULT.json and CPU_STATIC_CHECKS.json',blocker='Original complete API lacks independent FP32 row/K128 scale mode; expected remote vendor paths missing',next_action='acceptance_root review; no guarded check eligible',npu_runs=0))
print('PASS: 15 exact contracts; 30 direct/swapped mismatch checks; no imports/device runs')
