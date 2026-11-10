from pathlib import Path
import json,shutil,hashlib,datetime
W=Path('/root/dyq/workspace'); R=Path(__file__).resolve().parent; old=R.parent/'private'; scope=json.loads((R.parent/'SCOPE.json').read_text())
(R/'STARTED.json').write_text(json.dumps({'host':'local','scope_count':5,'owner':'selected-integration-20261026/nn_misc_math_recovery01','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'planned_original_apis':{'SumFwdOp':'ReduceSum','BatchNormBwdOp':'BatchNormGradV3','CountNonzeroFwdOp':'LpNormV2(p=0)','SoftmaxFwdOp':'SoftmaxV2 (historical success skipped)','VectorNormFwdOp':'LpNormV2(p=inf)'}},indent=2))
for pkg in ['ops_nn','ops_math']:
 shutil.copytree(old/'benchmarks/ascend_all_pairs'/pkg,R/'providers'/pkg,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
packet=W/'.cache/all-pairs/full_rollout_20261024/local24/group1/resume25_common_repair/continuation0320/tasks/current-original-nn-norm-launch-repair20/packet'
shutil.copy2(packet/'native_log_capture.py',R/'native_log_capture.py')
# Checker remains byte-identical and uses the real canonical root. No copied oracle.
(R/'CHECKER_PIN.json').write_text(json.dumps({'path':str(W/'TileOPs/scripts/ascend_pair_check.py'),'sha256':hashlib.sha256((W/'TileOPs/scripts/ascend_pair_check.py').read_bytes()).hexdigest()},indent=2))
