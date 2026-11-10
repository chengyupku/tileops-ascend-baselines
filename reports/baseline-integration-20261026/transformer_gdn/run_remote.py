from pathlib import Path
import json,subprocess,datetime,sys
W=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-20261026/transformer_gdn');sys.path.insert(0,str(W/'chunk_package'))
from resource_gate import require_capacity
contracts=json.loads((W/'CONTRACTS.json').read_text());est=json.loads((W/'CASE_RESOURCE_ESTIMATES.json').read_text())
cases=sorted(contracts,key=lambda i:(contracts[i]['op']=='DeltaNetFwdOp',est[i]['declared_tensor_bytes']))
blocked={}
for i in cases:
 d=W/i;route='delta' if contracts[i]['op']=='DeltaNetFwdOp' else ('recurrent' if 'decode' in i else 'chunk');pkg=W/('recurrent_package' if route=='recurrent' else 'chunk_package')
 if (d/'GUARD_RECEIPT.json').exists():continue
 if route in blocked:
  (d/'blocked.json').write_text(json.dumps({'reason':'Shared route blocker; not retried','evidence':blocked[route]}));continue
 try:require_capacity(pkg,d)
 except Exception as e:
  (d/'blocked.json').write_text(json.dumps({'reason':'Resource admission unavailable','error':str(e)}));print(i,'resource_blocked',flush=True);continue
 argv=['npu-guard','run','--owner','selected-integration-20261026/transformer_gdn','--wait','3600','--timeout','300','--','bash',str(d/'child.sh')]
 (d/'ARGV.json').write_text(json.dumps(argv));start=datetime.datetime.now(datetime.timezone.utc).isoformat();print('RUN',i,flush=True)
 with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
 (d/'GUARD_RECEIPT.json').write_text(json.dumps({'argv':argv,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode},indent=2));print('DONE',i,p.returncode,flush=True)
 if p.returncode!=0:blocked[route]=str(d)
