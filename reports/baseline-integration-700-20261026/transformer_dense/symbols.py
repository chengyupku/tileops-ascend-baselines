import json,subprocess,pathlib
base=pathlib.Path('/opt/tileops-all-pairs/workspace');paths=[base/'.cache/all-pairs/vllm/install/vendors/custom_transformer',base/'.cache/all-pairs/full_rollout_20261024/transfers/shared_fla_arm_1614/runtime/fla_npu/opp/vendors/fla_npu_transformer',base/'.cache/all-pairs/full_rollout_20261024/transfers/shared_ops_transformer_chunk_1752/runtime/vendors/linear_recovery_transformer'];out=[]
for p in paths:
 lib=p/'op_api/lib/libcust_opapi.so';r=subprocess.run(['nm','-D','--defined-only',str(lib)],capture_output=True,text=True);syms=[l.split()[-1] for l in r.stdout.splitlines() if ' aclnn' in l];out.append(dict(path=str(p),api_symbols=syms,device_configs=[str(x.relative_to(p)) for x in p.rglob('*') if x.suffix=='.json' and 'config' in str(x)]))
print(json.dumps(out,indent=2))
