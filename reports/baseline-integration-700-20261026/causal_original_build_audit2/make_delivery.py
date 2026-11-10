from pathlib import Path
import json,hashlib,datetime
p=Path(__file__).parent
allfiles=[]
for name in ['SOURCE_SNAPSHOT.json','GENERATED_AND_SDK_SNAPSHOT.json','DEPLOYMENT_SNAPSHOT.json','BUILD_CHAIN_SNAPSHOT.json','DEPENDENCIES_SNAPSHOT.json']:
 x=json.loads((p/name).read_text()); allfiles+=x if isinstance(x,list) else x['files']
by={f['path'].split('/source/')[-1]:f for f in allfiles}
def ref(rel,a,b):
 f=by[rel];return dict(path=f['path'],sha256=f['sha256'],lines=[a,b],excerpt='\n'.join(f['text'].splitlines()[a-1:b]))
scope=json.loads((p/'SCOPE.json').read_text())
keys=['|'.join(r[k] for k in ['baseline','operator','case_id']) for r in scope]
evidence={
 'scope_exactkeys':keys,'host':'openlibing-DevEnv-055409','kind':'CPU static source/build-chain audit, not numerical validation',
 'finding':'Original ordinary package build has a concrete generation and deployment path for the same aclnnCausalConv1dUpdate API. Prior --jit package intentionally omitted binary/config targets. Actual compilation and numerical success remain unproven.',
 'chain':[
 {'step':'Public whole Update API invokes CausalConv1d and returns y plus in-place convStatesRef','evidence':ref('mamba/causal_conv1d/op_api/aclnn_causal_conv1d_update.cpp',339,386)},
 {'step':'Schema supports FP16/BF16, both outputs, ascend950, opFile causal_conv1d_apt','evidence':ref('mamba/causal_conv1d/op_host/causal_conv1d_def.cpp',24,91)},
 {'step':'--jit maps ENABLE_OPS_KERNEL=OFF; normal custom --pkg maps ON','evidence':ref('build.sh',2490,2504)},
 {'step':'Generated adaptation and commands exist even with JIT, kernel/config targets require ENABLE_OPS_KERNEL','evidence':ref('cmake/custom_build.cmake',1237,1275)},
 {'step':'Original op_build consumes op_host_aclnnExc to generate exc/aic-ascend950-ops-info.ini','evidence':ref('cmake/custom_build.cmake',910,927)},
 {'step':'INI -> ascendc_bin_param_build -> dtype parameter JSON and original asc_opc dynamic commands','evidence':ref('cmake/func.cmake',393,440)},
 {'step':'Compiler produces per-variant supportInfo JSON and objects; config target depends on all selected operator targets','evidence':ref('cmake/func.cmake',941,1013)},
 {'step':'Original aggregator consumes binFileName/supportInfo, writes causal_conv1d_apt.json and binary_info_config.json; no manual config allowed','evidence':ref('cmake/scripts/util/ascendc_ops_config.py',181,261)},
 {'step':'Custom package kernel/config destinations','evidence':ref('cmake/func.cmake',754,765)},
 {'step':'Custom install sets ASCEND_CUSTOM_OPP_PATH to vendor root and LD_LIBRARY_PATH to vendor op_api/lib','evidence':ref('cmake/scripts/custom/install.sh',297,315)}],
 'generated_scripts':[dict(path=f['path'],sha256=f['sha256'],command_lines=[dict(line=i,text=l) for i,l in enumerate(f['text'].splitlines(),1) if 'res=$(asc_opc' in l]) for f in allfiles if '/gen/CausalConv1d-' in f['path'] and f['path'].endswith('.sh')],
 'runtime_layout':{'vendor_root':'<private-install>/vendors/causalpairs_transformer','env':'ASCEND_CUSTOM_OPP_PATH=<vendor_root>; LD_LIBRARY_PATH=<vendor_root>/op_api/lib:<matching-sdk>/lib64:...','config':'<vendor_root>/op_impl/ai_core/tbe/kernel/config/ascend950/causal_conv1d_apt.json','summary':'<vendor_root>/op_impl/ai_core/tbe/kernel/config/ascend950/binary_info_config.json','binaries':'<vendor_root>/op_impl/ai_core/tbe/kernel/ascend950/causal_conv1d_apt/CausalConv1d_<hash>.{o,json}','naming_note':'Update is the public API, CausalConv1d is the runtime op type, causal_conv1d_apt is the schema opFile. Do not search only for CausalConv1dUpdate.json. Source op_host/config/ascend950/causal_conv1d_binary.json is not the runtime aggregate config and must not be renamed/copied as it.'},
 'sdk_evidence':{'present':'9.1 asc-devkit metadata, asc_opc, bisheng, asc_op_compile_base/asc_op_compiler/compile_op.py, native protoc cache, opbase, catlass, ops-tensor, CMake cached master-058 source','not_blocking_by_itself':'Legacy cached ASCEND_CMAKE_DIR ends /tikcpp/ascendc_kernel_cmake/cmake and does not exist. Current source uses its own util scripts; actual generated wrapper imports asc_op_compile_base, whose files exist. No new compiler error observed. Do not declare unsupported from this stale legacy path.','version_requirement':ref('version.cmake',11,20)},
 'offline_staging_cautions':[ref('cmake/fetch_cann_cmake.cmake',11,36),ref('cmake/third_party/ops-tensor.cmake',12,26)],
 'full_api_boundary':{'entrypoints':['aclnnCausalConv1dUpdateGetWorkspaceSize','aclnnCausalConv1dUpdate'],'required_check_outputs_per_case':['output/y','complete updated conv_state'], 'required_check_outputs_total':4,'adaptation':'Retain existing whole binding layout transpose and full history/state copyback; no replacement convolution, oracle edit, state truncation or kernel change. API source SHA matches the prior causal_checker.json.','unknown':'Header exposes optional scheduling fields; do not infer arbitrary-option coverage from these two cases. Only exactkeys above are in scope.'},
 'prior_runtime_error':'Prior checker: aclnnCausalConv1dUpdateGetWorkspaceSize returned 561002; no automatic retry. Prior task attributes it to ParseDynamicKernelConfig. This audit did not rerun the failure.',
 'source_manifest':[dict(path=f['path'],sha256=f['sha256']) for f in allfiles],
 'accepted_added':0}
(p/'CONTRACT_OR_BUILD_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
preflight='''#!/usr/bin/env python3
# Read-only CPU preflight: stdlib only; never import torch, ACL, or vendor Python.
import pathlib,sys,json,hashlib
src=pathlib.Path(sys.argv[1]); sdk=pathlib.Path(sys.argv[2])
required_source=['build.sh','cmake/func.cmake','cmake/scripts/util/ascendc_ops_config.py','cmake/scripts/util/ascendc_bin_param_build.py','mamba/causal_conv1d/op_api/aclnn_causal_conv1d_update.cpp','mamba/causal_conv1d/op_host/causal_conv1d_def.cpp','mamba/causal_conv1d/op_kernel/causal_conv1d_apt.cpp','third_party/lib_cache/protoc/protoc','third_party/opbase','third_party/ops-tensor','third_party/catlass']
required_sdk=['bin/asc_opc','tools/opbuild/op_build','tools/bisheng_compiler/bin/bisheng','python/site-packages/asc_op_compile_base/asc_op_compiler/compile_op.py','aarch64-linux/asc/include/kernel_operator.h','share/info/asc-devkit/version.info','lib64/libnnopbase.so']
checks=[{'path':str(root/rel),'exists':(root/rel).exists()} for root,rels in [(src,required_source),(sdk,required_sdk)] for rel in rels]
cache=[src/'third_party/cann-cmake/function/prepare.cmake',src/'build/_deps/cann-cmake-src/function/prepare.cmake']
checks.append({'path':'local cann-cmake source (one of '+', '.join(map(str,cache))+')','exists':any(f.is_file() for f in cache)})
api=src/required_source[4]
print(json.dumps({'checks':checks,'api_sha256':hashlib.sha256(api.read_bytes()).hexdigest() if api.is_file() else None,'scope':'presence/hash only, not compiler/runtime validation'},indent=2))
sys.exit(0 if all(c['exists'] for c in checks) else 1)
'''
(p/'cpu_preflight.py').write_text(preflight)
recipe='''# FUTURE ACTION ONLY — not executed by this CPU audit.
# On server2, parent first selects its own writable deployment directory.
# Make a full independent copy of original source and third_party, excluding stale build/output.
# Before excluding build, retain build/_deps/cann-cmake-src unchanged under
# <copy>/third_party/cann-cmake (original CMake supports this offline source).
# Place unchanged copied third_party/ops-tensor at <copy>/../ops-tensor to use
# the original sibling branch and avoid its unconditional git fetch branch.
# Preserve third_party/pkg and lib_cache, native protoc and vendored headers.
# Check pinned dependency revisions; never point mutation-prone CMake git checkout
# at shared repositories. No edits to original kernel/API/compiler/oracle.
# Run this CPU presence preflight against original source first and copied source next:
python3 cpu_preflight.py <copy> /usr/local/Ascend/cann-9.1.0

# In a private child script; preserve original BUILD_RECIPES.txt offline packaging
# Python paths and cache dependencies, with the following essential SDK correction:
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export PATH="/opt/tileops-ci/runtime/bin:$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:/usr/bin:/bin"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 PYTHONDONTWRITEBYTECODE=1
export PIP_NO_INDEX=1 CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2
# Select only copied build_deps/native SDK Python directories; do not run imports as preflight.
# Allocate private TMPDIR, compiler caches and log directories; retain matching SDK loader paths.
cd <copy>
bash build.sh --pkg --soc=ascend950 --ops=causal_conv1d --vendor_name=causalpairs -j2 --build-type=Release --ccache off

# Entire original build child script above MUST be foreground under host guard:
# npu-guard run --owner 'root/causal-original-normal-build' --wait 600 --timeout 900 -- bash <private-build-child.sh>
# op_build/prepare, asc_opc and compiler Python may load runtime/driver: guard from
# entry, not just the final kernel. Do not use --jit or fabricate JSON.
# Budget suggestion (unmeasured): CPU staging/preflight 1–3 min; guarded original
# two dtype build/packaging 5–15 min (900 s cap), then 120 s separately guarded
# complete two-case checker. A timeout is an unresolved build, not unsupported API.
# Extract/install successful package into private destination using the original
# package installer --install-path=<private-install>; preserve all package files.
# Verify host API/tiling/config and ALL binary references before any device retry.
# Set vendor-root ASCEND_CUSTOM_OPP_PATH and matching op_api/lib loader path.
# Parent runs canonical checker for both SCOPE exactkeys and both output/state
# outputs; numerical acceptance requires four complete output checks.
'''
(p/'BUILD_NEXT_STEPS.txt').write_text(recipe)
rows=[]
for r in scope:
 rows.append(dict(exactkey='|'.join(r[k] for k in ['baseline','operator','case_id']),baseline=r['baseline'],operator=r['operator'],case_id=r['case_id'],status='deployment_route_identified_pending_original_build',api='aclnnCausalConv1dUpdate',reason=evidence['finding'],accepted_added=0,checked_outputs=0,total_outputs=2,registered=False))
now=datetime.datetime.now(datetime.timezone.utc)
result=dict(at=now.isoformat(),rows=rows,accepted_added=0,numerical_success=False,compile_executed=False,npu_calls=0,forbidden_imports=0,downloads=0,scope_expanded=False,shared_mutations=False,route_found=True,definitive_source_sdk_blocker=False,remaining_issues=['Normal unchanged two-variant build and complete private package closure not executed.','No measured build-time estimate or numerical result.','Keep runtime matching the package; canonical checker must validate y and full updated state on both exactkeys.','Preflight checks only file existence and selected hashes, not ABI compatibility or compilation.'],evidence='CONTRACT_OR_BUILD_EVIDENCE.json',next_step='BUILD_NEXT_STEPS.txt')
(p/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
start=datetime.datetime.fromtimestamp((p/'SPAWN.json').stat().st_mtime,datetime.timezone.utc)
delivery=dict(worker='1ce76fb498e44fc2',at=now.isoformat(),status='CPU audit complete: original normal package route identified',accepted_added=0,exactkeys=keys,ownership='root retained; no registry/provider/memory changes',files=[f.name for f in p.iterdir() if f.name not in ['SCOPE.json','SPAWN.json','TASK.txt']],checks=['original build/CMake/schema/config generation chain traced','two existing original asc_opc dtype commands inspected without execution','SDK/compiler/module/native protoc presence checked through file reads','Update API SHA matched prior checker','JSON artifacts and stdlib preflight syntax checked in final local check'],accounting={'diagnostic_shell_invocations_before_artifact_creation':20,'readonly_ssh_invocations':9,'artifact_creation_shell_invocations':1,'final_validation_shell_invocations_planned':1,'wall_seconds_since_supervisor_spawn_file':round((now-start).total_seconds(),1),'compile_seconds':0,'device_seconds':0},remaining_issues=result['remaining_issues'])
(p/'DELIVERY.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'created':['DELIVERY.json','RESULT.json','CONTRACT_OR_BUILD_EVIDENCE.json','cpu_preflight.py','BUILD_NEXT_STEPS.txt'],'elapsed_seconds':delivery['accounting']['wall_seconds_since_supervisor_spawn_file']}))
