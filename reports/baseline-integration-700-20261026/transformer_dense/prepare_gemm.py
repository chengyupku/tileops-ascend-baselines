import pathlib,json,shutil,hashlib,os
p=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense');g=p/'gemm';old='/opt/tileops-all-pairs/workspace/.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group2/resume_20261025/tasks/pristine_gemm_build';deps=g/'private_provider/op_impl/ai_core/tbe/recovery1217_gemm_transformer_impl/ascendc/common/tensor_api'
if not deps.with_name('tensor_api_sdk91_backup').exists():deps.rename(deps.with_name('tensor_api_sdk91_backup'));shutil.copytree(p/'tensor_api_fd51475',deps)
cmd=json.loads((g/'COMMANDS.json').read_text())[0]['argv'];cmd=cmd[cmd.index('--')+1:];cmd=[x.replace(old,str(g)) for x in cmd]
(g/'run').mkdir(exist_ok=True);(g/'package/33').mkdir(parents=True,exist_ok=True)
# Reuse original environment, relocate this exact target's paths only. No compiler flag changes.
e=json.loads((g/'ENVIRONMENT.json').read_text());e={k:v.replace(old,str(g)) for k,v in e.items() if isinstance(v,str)}
e['HI_PYTHON']='/opt/tileops-ci/runtime/bin/python';e['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';e['FLA_NPU_DISABLE_PTH']='1'
(g/'build_command.json').write_text(json.dumps(cmd,indent=2));(g/'build_env.json').write_text(json.dumps(e,indent=2))
h={str(f.relative_to(g)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (g/'private_provider').rglob('*.py')};(g/'wrapper_hashes.json').write_text(json.dumps(h,indent=2))
print(json.dumps({'command':cmd,'dependency':str(deps),'wrappers':h},indent=2))
