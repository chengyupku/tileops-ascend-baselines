"""Assemble CPU-only delivery evidence; writes only this directory."""
from pathlib import Path
import datetime
import hashlib
import json
D=Path(__file__).resolve().parent; W=D.parents[2]; R=D.parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(n,x): (D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
contracts=json.loads((D/'CONTRACTS.json').read_text())
checks=json.loads((D/'CPU_CHECKS.json').read_text())
assert checks['status']=='pass' and checks['count']==14
sources=[
'TileOPs/src/tileops/manifest/spec/gemm_bmm.yaml',
'TileOPs/src/tileops/manifest/spec/gemm.yaml',
'TileOPs/src/tileops/manifest/spec/convolution.yaml',
'TileOPs/workloads/gemm.py','TileOPs/workloads/convolution.py',
'TileOPs/workloads/workload_base.py','TileOPs/workloads/numerics.py',
'ascend_baselines/ops-transformer/gmm/grouped_matmul/op_api/aclnn_grouped_matmul_v5.h',
'reports/baseline-integration-700-20261026/gmm_fp16_remaining11/remaining_binding.py',
'reports/baseline-integration-700-20261026/gmm_fp16_remaining11/v5_fp32_bias.py',
'reports/baseline-integration-700-20261026/bf16_other_contract20/PARENT_REVIEW.json',
'reports/baseline-integration-700-20261026/bf16_other_contract20/SOURCE_MAPPING.json',
'reports/baseline-integration-700-20261026/bf16_other_contract20/CONTRACT_PLAN.json']
source_hashes={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in sources}
# The deployer is active. Deliberately do NOT publish its binding.py as a frozen pin.
remaining=[
'Parent release of local V5 deployment and final reviewed Python/native dependency hashes; bf16_local_gemm18 remains exclusive build owner. No install/provider binary is pinned by this delivery.',
'Parent must attest injected factory uses complete original aclnnGroupedMatmulV5GetWorkspaceSize/aclnnGroupedMatmulV5 and native optional bias; explicit Python factory hash alone is not transitive native provenance.',
'Actual torch packing/casts, native ABI/workspace/lifetime/device execution, numerical BF16 results, memory budget and original full-checker receipts remain unproved.',
'All 10 keys remain unrun/family-held. FP16 structural success and this mock cannot establish BF16 acceptance.',
'TN/TT Grouped transpose_a branches are deliberately rejected; no alternative algorithm provided.']
plan={
'execute_now':False,'execution_owner':'root','accepted_added':0,
'prerequisites':['Parent explicitly releases reviewed samehost native factory path/hash and native install manifest; no duplicate build.',
'Parent reviews adapter.py and CONTRACTS.json, then explicitly authorizes execution ownership; this worker has none.',
'Load path via adapter.load_factory(path,sha256=parent_approved_hash), or pass parent-audited dense_fp32 callable; no fallback.',
'Wrap complete canonical process including imports in npu-guard; synchronize async work before process exits.'],
'first_keys':[
['ops-transformer','BmmFwdOp','square-256-bfloat16'],
['ops-transformer','GroupedGemmFwdOp','qwen3-30b-ep8-empty-bfloat16'],
['ops-transformer','Conv1dFwdOp','whisper-conv1-bias-bfloat16']],
'follow_on':'Only after independent family full-checker evidence: remaining Bmm6 and Grouped moe-fwd. Preserve exact key inputs/seed/reference/checker/tolerances; do not repeat successful cases or reinterpret cost timeouts as BF16 numerical failure.',
'guard_template':'npu-guard run --owner root/bf16-scope10-<exact-key> --wait 600 --timeout <parent-budget> -- <samehost-original-fullchecker-command>',
'checker':'Original unchanged Bmm/Grouped WorkloadBase Exact BF16 .016/.016 over every output; Conv1d original dynamic convolution_verification including input-dependent atol and normalized bound. Original input generation, optional zero bias, seed and metadata retained.',
'acceptance':'Save exact native provenance/guard/checker receipts, complete shape/dtype/full-output and input-preservation results. Only parent may accept/register after review; no registration by this preparation.',
'cost':'Parent assigns time/memory budget for full original CPU references, particularly large Bmm and Grouped moe-fwd; no CPU reference diagnostic or numerical replay here.'}
save('RUNTIME_PLAN.json',plan)
result={'at':now(),'worker':'55516fb10a894c6c','status':'completed_cpu_interface_preparation_pending_parent_review',
'scope_count':10,'accepted_added':0,'execution_ownership':False,'registered':False,
'npu_runs':0,'build_runs':0,'reference_runs':0,'numeric_replays':0,'subagents':0,
'shared_provider_changes':0,'memory_changes':0,'source_modifications':0,
'cpu_checks_passed':checks['count'],'numerical_validation':False,
'findings':['All params including Conv1d T and bias presence resolved from exact manifest workload and dtype_cases, checked against scoped audit.',
'Bmm passes complete batch to original TensorList branch; intercepted original V5 control-flow mock verifies all batch entries in order.',
'Conv1d retains original windows/permute/reshape and native optional bias; original dynamic checker retained unchanged.',
'Grouped validates canonical prefix offsets, skips zero-row NN/NT groups and scatters all live rows to original positions. Tests include 8 live + 8 empty, interspersed empties, all-empty and invalid offsets.',
'Original V5 source mock checks 20 argument positions, lossless-widening/cast operation order and synchronize-before-destroy; does not execute native ABI or any numerical product.',
'failed_reference_bias_confirmed applies to Gemm, not these Conv1d/Bmm/Grouped exact keys.'],
'per_key':[{'key':r['key'],'status':'unrun_family_hold','cpu_mock':r['cpu_mock'],'accepted_added':0} for r in contracts['rows']],
'remaining_issues':remaining,
'read_efficiency_note':'Initial filename discovery under memory was overly broad and output truncated; no historical file contents from that discovery were loaded. Subsequent task memory read limited to integration_700_20261026. No memory writes.'}
save('RESULT.json',result)
save('DELIVERY.json',{'at':now(),'worker':result['worker'],'status':result['status'],'accepted_added':0,'execution_ownership':False,
'entrypoints':{'adapter.py:make_runner':'Explicit factory + optional fake tensor API; returns (run, native candidate), preserving native cleanup ownership.',
'adapter.py:load_factory':'Explicit path and required matching SHA256; isolated module object, no fixed sys.modules alias.',
'adapter.py:candidate':'Future parent wrapper returning replaced Candidate with native cleanup/source provenance; no registration.'},
'artifacts':['adapter.py','cpu_checks.py','CPU_CHECKS.json','CONTRACTS.json','RUNTIME_PLAN.json','FILEHASHES.json','RESULT.json','DELIVERY.json','HEARTBEAT.json','finalize.py'],
'checks_command':'PYTHONDONTWRITEBYTECODE=1 python3 '+str(D/'cpu_checks.py'),
'checks_passed':checks['count'],'numerical_validation':False,'source_boundary':'Only thin interface/copy/packing/transpose/scatter/output dtype checks. Original kernel/compiler/architecture/reference/checker/seed/tolerances untouched; gemm_cpu_bf16.patch not applied.',
'pending_native_pin':'Required after bf16_local_gemm18 deployer and parent release; running binding.py was read only as an interface example, not frozen.',
'remaining_issues':remaining})
save('HEARTBEAT.json',{'at':now(),'phase':'completed','accepted_added':0,'npu_runs':0,'cpu_checks_passed':checks['count']})
artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.iterdir() if p.is_file() and p.name!='FILEHASHES.json'}
save('FILEHASHES.json',{'at':now(),'algorithm':'sha256','artifacts':artifacts,'read_sources':source_hashes,
'boundary':'Observed source hashes at delivery, not proof of source immutability before observation. FILEHASHES excludes itself. No native binaries or mutable deployer binding pinned; parent must supply final full provenance.'})
print(json.dumps({'delivery':str(D/'DELIVERY.json'),'artifacts_hashed':len(artifacts),'source_hashes':len(source_hashes),'accepted_added':0}))
