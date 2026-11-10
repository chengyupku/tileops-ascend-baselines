from pathlib import Path
import json,hashlib,shutil
R=Path('/opt/tileops-all-pairs/workspace');W=R/'reports/baseline-integration-700-20261026/transformer_linear';T=R/'.cache/all-pairs/full_rollout_20261024/remote_ready/server2/group1/resume_20261025/continuation0430/tasks';E=W/'prior_evidence';E.mkdir(exist_ok=True)
files=[T/'recovery0730-gdn-fixed-abi/FP16_DEPLOYMENT_BLOCKER.json',T/'recovery0730-delta-whole-api-deployment/ORACLE_DEPLOYMENT_DIAGNOSIS.json',T/'recovery0730-delta-whole-api-deployment/CHUNK_BUILD_GUARD.json',T/'recovery0730-delta-whole-api-deployment/chunk-build.stderr.log',T/'recovery0730-delta-whole-api-deployment/chunk-build.stdout.log']
for f in files:
 if f.exists():shutil.copyfile(f,E/f.name)
found=[]
for root in [T,R/'.cache/ops-transformer',R/'ascend_baselines/ops-transformer']:
 if root.exists():
  for f in root.rglob('*'):
   if f.is_file() and ('causal' in f.name.lower() or 'chunk_state' in f.name.lower() or 'chunk_kda' in f.name.lower()) and (f.suffix in ['.so','.json'] or f.name.endswith('.so.1')):found.append(str(f))
(W/'BINARY_SEARCH.json').write_text(json.dumps(found,indent=2));print('\n'.join(found)[:8000])
old=json.loads((T/'recovery0730-delta-whole-api-deployment/ORACLE_DEPLOYMENT_DIAGNOSIS.json').read_text());obs=[dict(path=x['path'],exists=Path(x['path']).exists()) for x in old['sdk_file_observations']];(W/'ORACLE_CURRENT.json').write_text(json.dumps(obs,indent=2));print('metadata',obs)
