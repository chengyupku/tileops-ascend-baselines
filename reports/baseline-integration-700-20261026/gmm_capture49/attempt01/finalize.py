from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
P=Path(__file__).resolve().parent;T=P.parent;E=P/'remote_evidence'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(E/'REMOTE_ARTIFACTS.json');assert all(sha(E/k)==v for k,v in manifest.items())
r=read(E/'representative.json');c=read(E/'REFERENCE_COMPARISON.json');causal=read(P/'CAUSAL_COMPARISON.json');dispatch=read(E/'NATIVE_DISPATCH.json');cap=read(E/'CAPTURE.json');scope=read(T/'SCOPE.json')
assert r['input_seed']==1311006047 and r['status']=='failed' and r['stage']=='verify'
assert not r['oracle_source_changed'] and not r['provider_source_changed']
assert read(E/'EXECUTION.json')['frozen_unchanged']
assert (E/'guard.exit').read_text().strip()=='1'
assert len(scope)==49
now=datetime.now(timezone.utc).isoformat()
write(P/'VALIDATION.json',{'utc':now,'remote_artifacts_verified':len(manifest),'copied_hash_mismatches':0,'original_checker_unmodified':True,'original_oracle_unmodified':True,'frozen_before_and_after_equal':True,'native_original_binary_observed':dispatch['selected_binary'],'new_seed':r['input_seed'],'captured_calls':read(E/'EXECUTION.json')['capture_counts'],'source_and_shared_provider_changes':False,'cpu_runtime_npu_imported':c['torch_npu_imported'],'cpu_runtime_ascend_libraries':c['loaded_ascend_libraries']})
rows=[]
for item in scope:
 key=[item['baseline'],item['operator'],item['case_id']];representative=key==['ops-transformer','GemmFwdOp','square-1k-bfloat16']
 rows.append({'utc':now,'key':key,'status':'failed_reference_bias_confirmed' if representative else 'unresolved_not_run','registered':False,'verified_unregistered':False,'accepted':False,'npu_runs':int(representative),'checker_exit_code':1 if representative else None,'reason':'New exact canonical input establishes original reference deviation; original native output within original tolerance of FP64. Cannot accept without oracle change, which is prohibited.' if representative else 'No direct experiment on this key; representative evidence does not establish sibling numerical failure or impossibility. No proven legal binding repair to authorize expansion.','checker_receipt':'attempt01/remote_evidence/representative.json' if representative else None,'api_queue_s':None})
(T/'RESULT.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows));write(T/'RESULT.json',rows)
write(P/'HEARTBEAT.json',{'utc':now,'phase':'complete','npu_runs':1,'api_queue_s':None})
(T/'FINDINGS.md').write_text('''# Canonical captured-input diagnosis

One new canonical execution only: ops-transformer / GemmFwdOp / square-1k-bfloat16, full 1024×1024 BF16, seed 1311006047. Original complete checker, original CPU reference and original complete GroupedMatmulV5 binding/API were used. The private observation wrapper copied inputs/outputs and delegated to the original bodies; it made no arithmetic, tolerance, seed, shape, kernel, schedule or checker-result changes.

The checker failed at verify with exit code **1**, **30,312 / 1,048,576** mismatches. All one output and both inputs were retained. CPU reference inputs, native NPU inputs, and each call's post-inputs are byte-identical. Source and provider hashes remained unchanged. No new canonical success, registration or acceptance: **0 verified_unregistered, 0 accepted**.

Same saved tensors, CPU-only analysis outside the NPU reservation:

| Actual → target | Violations at original 0.016 absolute/relative tolerance (evaluated in FP64) | Unequal BF16 values |
|---|---:|---:|
| Original native → captured original reference | 30,312 | 428,372 |
| FP64 rounded BF16 → captured original reference | 30,309 | 428,370 |
| FP32 rounded BF16 → captured original reference | 30,311 | 428,364 |
| Original native → FP64 rounded BF16 | 0 | 118 |
| FP32 rounded BF16 → FP64 rounded BF16 | 0 | 147 |

All 30,309 FP64-vs-reference violations overlap the native failure set; native contributes 3 additional threshold crossings. The candidate is not bitwise equal to FP64, but all differences satisfy the unchanged tolerance. This directly establishes reference deviation for this new canonical input and explains almost the entire failing set. This representative cannot be accepted under the current unchanged-oracle contract. No candidate modification to imitate the reference and no oracle patch is proposed.

Native dispatch logs show original private_provider GroupedMatmul_0d2d43751f81d7f29646a862147967c9.o registered on device, SHA256 f0ae0de77f9b9448666e336a73432a9eb9898d0f181554c6f9d930972381a08a. V5 parameters retain splitItem=0, groupType=-1, groupListType=0, actType=0, transposeX=0, transposeWeight=0, groupList=null and outLength=1. No legal binding defect was found.

Numerical reporting caveats: CPU BF16 torch.isclose/assert_close gives 30,209 native-vs-reference and 30,206 FP64-vs-reference mismatches because threshold evaluation differs from the original NPU check. These CPU results are retained separately; the original NPU checker result remains authoritative. FP64 evaluation matches its 30,312 native mismatch count. Greatest absolute difference across *all* elements is 1.0; maximum among violating elements is 0.375, matching the original checker message. Seven NPZ files retain full per-element actual, target, signed/absolute error, tolerance limit and violation masks, plus summary quantiles; no outputs are omitted.

This is a new experiment, not recovered historical input. Matching historical mismatch counts do not establish byte identity of historical tensors. The other 48 exact keys were not run and cannot be classified as numerical failures or impossible from this result. No batch, benchmark, autotune, profiling, additional NPU probe, shared write, registry change or memory write occurred. API queue time remains null; local and remote wall clocks differ by roughly 82 seconds, so no cross-host timing is inferred.

Evidence: attempt01/remote_evidence/representative.json, CAPTURE.json and raw .bin files; FROZEN.json; EXECUTION.json; source_snapshot/; NATIVE_DISPATCH.json and complete ASCEND_PROCESS_LOG_PATH logs; REFERENCE_COMPARISON.json and all .npz distributions; attempt01/CAUSAL_COMPARISON.json; attempt01/VALIDATION.json. All 53 files in the sealed remote artifact manifest were downloaded and verified byte-for-byte. The supplementary causal comparison is separately hashed by DELIVERY.json.
''')
files={str(p.relative_to(T)):sha(p) for p in T.rglob('*') if p.is_file() and p.name not in ('DELIVERY.json',)}
write(T/'DELIVERY.json',{'utc':now,'worker':'5bd1685bac614114','status':'diagnostic_complete_unaccepted','scope_count':49,'npu_runs':1,'canonical_failures':1,'unrun':48,'verified_unregistered':0,'accepted':0,'registered':0,'checker_exit_code':1,'api_queue_s':None,'checker_wall_seconds':r['wall_seconds'],'patch':None,'binding_repair':None,'remote_private':'/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gmm_capture49/attempt01','findings':'FINDINGS.md','result_lines':'RESULT.jsonl','validation':'attempt01/VALIDATION.json','original_canonical_receipt':'attempt01/remote_evidence/representative.json','causal_comparison':'attempt01/CAUSAL_COMPARISON.json','native_binary':dispatch['selected_binary'],'remaining_issues':['Representative cannot be accepted without altering oracle contract; no oracle change authorized or made.','Other 48 keys remain untested; no generalization of impossibility.','Historical byte identity unavailable; captured data belongs to this new execution.','CPU BF16 threshold counts differ from authoritative NPU checker; both retained.'],'file_sha256':files})
print(json.dumps({'rows':len(rows),'verified_remote_files':len(manifest),'npu_runs':1,'accepted':0,'delivery_files':len(files)}))
