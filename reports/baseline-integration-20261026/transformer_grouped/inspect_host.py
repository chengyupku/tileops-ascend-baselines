"""Read-only CPU deployment and exact-key receipt inspection; no framework imports."""
import json,os,platform,hashlib
from pathlib import Path
W=Path('/opt/tileops-all-pairs/workspace')
P=W/'reports/baseline-integration-20261026/transformer_grouped'
keys={tuple(r['key']) for r in json.loads((P/'SCOPE.json').read_text())}
report={'host':'server1','architecture':platform.machine(),'source_exists':(W/'ascend_baselines/ops-transformer').exists(),'standard_vendor_exists':(W/'.cache/all-pairs/ops_transformer/install/vendors/allpairs_transformer').exists(),'original_gmm_files':[],'exact_receipts':[],'target_rows':[]}
def rows(d):
 if isinstance(d,dict):
  key=d.get('key') or ([d.get('library'),d.get('op'),d.get('case_id')])
  if isinstance(key,list) and all(isinstance(x,str) for x in key) and tuple(key) in keys: yield d
  for v in d.values():yield from rows(v)
 elif isinstance(d,list):
  for v in d:yield from rows(v)
for base,dirs,files in os.walk(W/'.cache/all-pairs'):
 dirs[:]=[d for d in dirs if d not in {'.git','__pycache__','site-packages','node_modules'}]
 for name in files:
  path=Path(base)/name
  if name in {'aclnn_grouped_matmul_v5.h','aclnn_grouped_matmul_add.h','grouped_matmul_apt.cpp','grouped_matmul_add.cpp'}:report['original_gmm_files'].append(str(path))
  if name=='TARGET_SETS.json' or name=='RESULT.json' or ('checker' in name and name.endswith('.json')):
   if 'full_rollout_20261024/remote_ready/server1' not in str(path):continue
   try:d=json.loads(path.read_text())
   except (ValueError,OSError):continue
   for row in rows(d):
    if name=='TARGET_SETS.json':report['target_rows'].append({'path':str(path),'row':row})
    elif row.get('status')=='verified':report['exact_receipts'].append({'path':str(path),'row':row})
report['device_commands']=0
(P/'HOST_INSPECTION.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
