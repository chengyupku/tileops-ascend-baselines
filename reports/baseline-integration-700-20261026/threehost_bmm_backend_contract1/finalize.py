import json,hashlib,subprocess,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
D=json.loads((P/'DIAGNOSTIC.json').read_text());V=json.loads((P/'ARTIFACT_VERIFICATION.json').read_text())
for k,v in V['files'].items():
 assert v['sealed_occurrences'] and all(h==v['sha256'] for _,h in v['sealed_occurrences'])
 assert D['captured_window_hashes'][k[:-4]]==v['sha256']
assert D['metrics']['native']['unequal_to_rounded_math64']==0
assert D['metrics']['disabled']['unequal_to_rounded_math64']==0
for phase in ['safety_before','safety_after']:
 for key in ['forbidden_modules','device_fds','forbidden_mappings']:assert not D[phase][key]
contract={
 'status':'not_admissible_as_unchanged_frozen_environment; parent_may_explicitly_approve_backend_variant', 'accepted_added':0,
 'setting':'torch.backends.mkldnn.enabled = False',
 'operation_contract':{'source':'STATIC_EVIDENCE.json; original private frozen workload gemm.py:364-387 and ascend_pair_check.py:137-161', 'expression':'torch.bmm(a,b), D[b,i,j]=sum_k A[b,i,k]*B[b,k,j]', 'inputs':'unchanged contiguous BF16 [4,4096,4096], same saved bytes', 'output':'BF16 [4,4096,4096]', 'tolerance':'rtol=atol=0.016; workloads/numerics.py:264-274,305-321', 'explicit_mkldnn_requirement_in_workload_or_wrapper':False, 'explicit_cpu_bf16_accumulator_mode_in_workload_or_wrapper':False},
 'public_api_evidence':{'file':'installed torch/backends/mkldnn/__init__.py:116 in STATIC_EVIDENCE.json', 'meaning':'Public ContextProp changes backend enabled state, not input dtype or torch.bmm expression. Backend arithmetic/order may change.', 'default':'Installed ATen/Context.h:485 enabled_mkldnn=true. Actual earlier diagnostic confirms enabled=True, deterministic=False, fp32_precision=none, float32_matmul_precision=highest.', 'accumulator':'Installed bmm docstring gives matrix product and shape, no universal CPU BF16 accumulation precision/order. FP32Precision controls and CUDA CuBLAS BF16 reduction controls do not establish a CPU BF16 accumulator switch.'},
 'frozen_environment_review':{'facts':['Original run.sh has no explicit mkldnn disable; captured reference used oneDNN gemm:acl with enabled=True.','FROZEN.json is a file hash closure, not an explicit authorization to vary numerical backend state.','No reviewed workload/launcher text explicitly mandates enabled=True as a mathematical rule. No reviewed contract grants permission to silently change it either.'], 'conflict':'enabled=False differs from the captured frozen execution state. It cannot be described as an unchanged-environment replay or silently replace the failed reference.', 'decision':'Not currently admissible for original frozen-environment acceptance. Semantically compatible backend variant is a concrete proposal for parent environment-policy decision; absence of a prohibition and public API status are insufficient approval.', 'limits':'This is bounded review of specified original sources, launcher, frozen closure and previous contract conclusion; not proof that every parent policy has been enumerated.'},
 'distinctions':{'backend_selection':'Same public BF16 bmm operation/dtype/output/tolerance, different implementation dispatch and possible rounding.', 'oracle_change':'a.float()/a.double() formal reference, custom fsum reference, alternative reduction formula or custom primitive would change algorithm/dtype and are prohibited here.', 'input_change':'Regenerating inputs from seed is not exact-byte replay; current diagnostic verifies saved SHA256 instead.'},
 'diagnostic_interpretation':'In one 32x32 complete-K window native and disabled outputs equal FP64 result rounded to BF16 for all1024 values. Default differs for612 and exceeds original tolerance against FP64 for76. Supports numerical backend discrepancy, not exact internal accumulator cause or global correctness.',
 'next_original_fullcase_plan':[
 'Parent first records whether a public CPU backend variant is allowed by baseline700 evaluation policy. If exact environment is immutable, retain failed status; do not run a disabled variant for acceptance.',
 'If explicitly admitted, use a new private fullcase report with backend setting logged before/inside/after reference. Keep original workload, reference callable torch.bmm, checker, candidate, dtype, shape, output and tolerance byte-for-byte unchanged. Apply only the public enabled=False process setting (restore in finally); do not wrap with a replacement FP32/FP64 oracle or use captured output as reference.',
 'Parent chooses one bounded guarded full original checker on server1 using the host npu-guard protocol, preflight memory/disk and frozen hashes. No NPU execution is authorized to this worker. Use same exact saved inputs if existing approved harness supports replay; otherwise record a new run and its byte hashes without claiming same inputs.',
 'Capture actual settings, library hashes, primitive logs, candidate API returns, input preservation and all 67108864 output comparisons. Window results are not sufficient. Budget CPU fallback time/memory explicitly; on timeout preserve evidence and do not repeat unchanged.',
 'Only a full original checker pass under the explicitly admitted runtime variant can be considered by parent for acceptance. Preserve the original enabled=True failure and disclose reference/backend timings; do not mix timings across environments.'
 ],
 'not_done':['No formal backend override applied','No FP32/FP64 formal oracle','No full67M CPU recomputation','No NPU/native candidate load/compile','No acceptance/ledger/provider/memory edits']}
(P/'CONTRACT_JUDGMENT.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
(P/'CONTRACT_JUDGMENT.md').write_text('''# BF16 BMM backend contract review

结论：`mkldnn.enabled=False` 保留原 BF16 `torch.bmm` 的表达式、输入/输出 dtype 和原容差，属于 CPU 执行后端配置；但它改变已捕获 `enabled=True` 的运行状态，**当前不能作为“冻结环境不变”的正式准入**。父 Agent 须明确批准环境变体后才可考虑完整原 checker。公开开关本身不构成授权。

原 workload 和 CPU wrapper 没有明确指定 CPU BF16 累加模式或要求 mkldnn=True。当地 Torch 的 Context.h 默认 enabled_mkldnn=true；原 launcher 未关闭它，同次运行已证实 oneDNN gemm:acl。冻结文件哈希不能证明允许后端状态变更。此结论不将“未发现强制条款”误当作自动准入，也不认定所有环境变体永久禁止。

一次 CPU 诊断：batch0、M/N[0,32)、K[0,4096)，读取原始输入并验证 SHA256；复用已封存的64×64 native/default/disabled输出，仅切出32×32，不重跑成功的 bmm 对照。用独立 Python FP64 乘法和 math.fsum 求1024个完整K点积。

|输出|与舍入到BF16的数学值不同|相对FP64超原容差|平均绝对误差|
|---|---:|---:|---:|
|captured native|0|0|0.07508369|
|disabled mkldnn|0|0|0.07508369|
|default / captured reference|612|76|0.19007925|

native/disabled 在612点更接近数学值，412点同距，没有点更差。BF16最终舍入自身最大绝对误差为0.48108717；原比较是 `abs(a-r) <= 0.016 + 0.016*abs(r)`，因此绝对误差大小不能单独判断失败。相对舍入后数学值，default有78点超容差。FP64仅是诊断证据，从未替换正式oracle。

一次进程exit0，-I -S -B，不加载.pth，autoload禁用；120s wall、8GiB AS、110s CPU。前后无torch_npu/Ascend ACL映射或设备句柄；CPU Torch/Arm Compute Library属于允许的CPU依赖，不是Ascend ACL。无NPU、编译或候选native库加载。256MiB输入留在server1。

全量正确性、禁用后端的全量运行成本、确切内部累加机制仍未知。下一步仅由父 Agent 决定：先明确环境准入，再在新私有目录、主机NPU锁和预算下运行一次原完整checker，保存所有67108864输出检查与环境证据；未准入则保留原失败。不重复窗口，不修改oracle/checker/tolerance。
''')
delivery={'worker':'2bdbe286ff454b2f','status':'completed_cpu_only_contract_and_single_diagnostic','completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_added':0,'diagnostic_runs':1,'diagnostic_exit':0,'contract_verdict':contract['status'],'key':['ops-transformer','BmmFwdOp','square-4k-bfloat16'],'findings':['No explicit mkldnn or CPU BF16 accumulation mandate in reviewed workload/wrapper; installed default and captured run enabled=True.','Public backend selection preserves mathematical operation/dtypes/tolerance but changes frozen execution state; no automatic admission.','1024/1024 native and disabled results equal FP64 math rounded to BF16; default differs612, outside original tolerance versus FP64=76.','Independent CPU math.fsum only; sealed prior output windows reused without rerunning bmm.'],'checks':['Exact saved input SHA256 matches prior capture','Four reused output windows match prior SEALED_HASHES','CPU isolation pre/post empty forbidden modules/mappings/device handles','One bounded CPU diagnostic exited0; no NPU/compiler/candidate-native loading','Only own local report and remote same-name private report written; no memory/ledger/oracle/kernel/checker/provider edits'],'remaining_issues':['Parent environment-policy admission required before backend variant formal check','Fullcase67108864 outputs unverified; accepted_added remains0','Exact backend accumulator mechanism and full fallback runtime unestablished'],'artifacts':['CONTRACT_JUDGMENT.json','CONTRACT_JUDGMENT.md','DIAGNOSTIC.json','WINDOW_VALUES.json','STATIC_EVIDENCE.json','ARTIFACT_VERIFICATION.json','FILEHASHES.json']}
(P/'DELIVERY.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(P.iterdir()) if p.is_file() and p.name!='FILEHASHES.json'}
(P/'FILEHASHES.json').write_text(json.dumps(hashes,indent=2))
for name,digest in hashes.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
print(json.dumps({'verified_local_files':len(hashes),'accepted_added':0,'diagnostic_runs':1}))
