import json,subprocess,datetime,sys,time
from pathlib import Path
W=Path(__file__).resolve().parent;sys.path.insert(0,str(W/'chunk_package'));from resource_gate import require_capacity
case=sys.argv[1];d=W/case
if (d/'GUARD_RECEIPT.json').exists():raise RuntimeError('No unchanged retry')
require_capacity(W/'chunk_package',d)
argv=['npu-guard','run','--owner','baseline700-20261026/transformer_linear','--wait','3600','--timeout','300','--','bash',str(d/'child.sh')]
t=time.time();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
v=dict(utc=start,phase='guard_wait_or_run',current_keys=[['ops-transformer','GDNFwdOp',case]],last_progress_at=start,blocked_reason=None);(W/'HEARTBEAT.json').write_text(json.dumps(v,indent=2))
with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
(W/case/'GUARD_RECEIPT.json').write_text(json.dumps(dict(argv=argv,start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,elapsed_seconds=time.time()-t),indent=2));print(case,p.returncode,flush=True)
