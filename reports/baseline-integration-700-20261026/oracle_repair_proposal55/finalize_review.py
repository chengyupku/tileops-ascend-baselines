import ast, hashlib, json, collections
from pathlib import Path
from datetime import datetime,timezone
W=Path('/root/dyq/workspace');P=Path(__file__).resolve().parent;R=W/'reports/baseline-integration-700-20261026'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(P/n).write_text(json.dumps(x,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
receipt=read(R/'sgl_gdn_remote38/runs/batch06--GDNFwdOp--prefill-narrow-state-bfloat16/checker.json')
checks={}
for f,h in receipt['oracle_source_sha256'].items():
 p=W/f
 checks[f]={'receipt':h,'current':sha(p),'match':h==sha(p)}
checks['TileOPs/scripts/ascend_pair_check.py']={'receipt':receipt['verifier_sha256'],'current':sha(W/'TileOPs/scripts/ascend_pair_check.py'),'match':receipt['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py')}
G=R/'gmm_capture49/attempt01/remote_evidence'
checks['GEMM_snapshot_vs_current']={'snapshot':sha(G/'source_snapshot/gemm.py'),'current':sha(W/'TileOPs/workloads/gemm.py'),'match':sha(G/'source_snapshot/gemm.py')==sha(W/'TileOPs/workloads/gemm.py')}
dump('RECEIPT_SOURCE_HASH_AUDIT.json',checks)
# CPU guard branch is the only changed AST statements; CUDA flags/try-finally exactly preserved.
a=ast.parse((W/'TileOPs/workloads/gemm.py').read_text());b=ast.parse((P/'gemm_cpu_bf16.py').read_text())
fa=next(x for x in a.body if isinstance(x,ast.FunctionDef) and x.name=='matmul_fp32_reduction');fb=next(x for x in b.body if isinstance(x,ast.FunctionDef) and x.name==fa.name)
del fb.body[1]
assert ast.dump(a)==ast.dump(b)
dump('AST_CHECKS.json',{'gemm_original_AST_identical_after_removing_CPU_branch':True,'gdn_math_file_unchanged':True,'numerics_file_unchanged':True,'gdn_patch_changes_checker_dispatch_only':True})
scope=read(P/'SCOPE.json'); counts=collections.Counter(x['baseline'] for x in scope)
proposal={'utc':datetime.now(timezone.utc).isoformat(),'status':'private proposals only; not authorized or applied','scope_file':str(P/'SCOPE.json'),'scope_sha256':sha(P/'SCOPE.json'),'scope_count':len(scope),'scope_by_baseline':dict(counts),'accepted_added':0,'registered':0,'npu_runs':0,'shared_source_writes':0,'proposals':[
 {'id':'gemm_cpu_bf16','patch':'gemm_cpu_bf16.patch','target':'TileOPs/workloads/gemm.py:matmul_fp32_reduction','original_sha256':read(P/'SOURCE_HASHES.json')['TileOPs/workloads/gemm.py'],'trigger':'both operands CPU BF16','before':'CUDA reduced precision flags do not constrain CPU BF16 matmul; captured aarch64 reference disagrees with saved FP64 result','after':'lossless widening BF16 -> FP32, FP32 matmul with CPU autocast disabled, narrow to original BF16; other device/dtype code AST identical','validation':'GEMM_CPU_VALIDATION.json','decision':'Ready for parent review, not acceptance','risks':['CPU BF16 consumers of this helper change numerical output and temporary memory/work; local x86 results do not test ARM FP32 backend','Only GemmWorkload calls this helper in current workloads tree. BmmWorkload uses torch.bmm; GroupedGemmWorkload uses torch.mm. This minimal patch does NOT establish a fix for all 49 scope keys.','FP32 accumulation is not exact real arithmetic. 117 BF16 values differ from saved FP64 rounding on this machine, all within unchanged tolerance; CUDA runtime untested but its original AST preserved.','CPU reduced precision behavior under other backend precision configurations needs review; no global backend settings modified.']},
 {'id':'gdn_cpu_dispatch','patch':'gdn_cpu_dispatch.patch','target':'TileOPs/scripts/ascend_pair_check.py:verify_gdn_prefill_on_cpu and main dispatch','original_sha256':read(P/'SOURCE_HASHES.json')['TileOPs/scripts/ascend_pair_check.py'],'trigger':'sgl-kernel-npu GDNFwdOp with input token axis >1','before':'original gdn_ref prefill uses FP64 and NPU rejects DT_DOUBLE before candidate invocation','after':'same reference/validator execute CPU; unchanged original candidate receives original device arguments/reset, both output and FP32 state copied losslessly to CPU; run_device explicitly cpu during reference and restored','validation':'GDN_CPU_VALIDATION.json','decision':'Scheduling patch reviewable; actual captured-input validation blocked by missing payloads in supplied tree','risks':['Supplied prefill receipt has candidate=[] and reference returned=false. It contains metadata, not raw actual inputs. No actual GDN CPU replay is claimed.','Synthetic 65-token smoke verifies original FP64 chunk formula, complete o+state validator, corruption rejection, original argument/reset routing and device restoration; it does not execute NPU API.','Moving same floating point formula to CPU can change backend rounding; unchanged tolerance remains required. No FP64 -> FP32 downgrade.','Copies, complete output/state storage, verify snapshots and FP64 reference add CPU time/RAM; full q35 shapes not executed, no 100 acceptance/hour promise.','Reference run_device is process-global, matching existing checker single-case execution; not suitable for concurrent threads changing it.']}],
 'hash_audit':'RECEIPT_SOURCE_HASH_AUDIT.json','limitations':['55 is affected scope, not 55 observed failures. 48 other GEMM-family cases and 5 other GDN prefill cases are not claimed passed.','No shared oracle authorization sought by this worker; parent decides whether to seek authorization.','No baseline candidate code changed and no ACCEPTED/registry/memory writes.'],'remaining_issues':['Parent must identify/provide existing actual GDN capture payload + hash manifest or authorize a later guarded capture by its NPU owner. This worker cannot create it.','Review source hash drift before any application; use patch against recorded source hashes only.','After authorization, independently verify relevant actual cases through unchanged canonical contracts; count successes only then.']}
dump('REVIEW_PROPOSAL.json',proposal)
dump('HEARTBEAT.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'review completed','npu_used':False,'feedback_exists':(P/'FEEDBACK.txt').exists()})
if (P/'FEEDBACK.txt').exists():print((P/'FEEDBACK.txt').read_text())
print('Hash anomalies:',json.dumps({k:v for k,v in checks.items() if not v['match']},indent=2))
print('Scope:',dict(counts))
