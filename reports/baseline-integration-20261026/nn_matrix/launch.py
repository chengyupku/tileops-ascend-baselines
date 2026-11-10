from pathlib import Path
import json,os,subprocess
P=Path(__file__).resolve().parent;W=Path('/root/dyq/workspace')
k=['ops-nn','GemmINT8W8A8FwdOp','llama3-70b-prefill-down-bfloat16']
policy=json.loads((W/'.cache/all-pairs/full_rollout_20261024/repairs/resume_20261025/boundary_1557/user_amnesty_0323/POLICY.json').read_text());assert k not in policy['amnesty_keys']
cmd=['npu-guard','run','--owner','selected-integration-20261026/nn_matrix','--wait','3600','--timeout','300','--',str(W/'.tools/baselines/baseline-run'),'ops-nn','python',str(P/'run_case.py')]
(P/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['FLA_NPU_DISABLE_PTH']='1'
with (P/'guard.log').open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=P)
(P/'GUARD_RESULT.json').write_text(json.dumps({'returncode':r.returncode}));raise SystemExit(r.returncode)
