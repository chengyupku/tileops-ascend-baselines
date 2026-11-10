import json,sys,subprocess,datetime,time,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;case='q35-gva-decode-b512-bfloat16';d=W/case
sys.path.insert(0,str(W/'recurrent_package'));from resource_gate import require_capacity
assert not (d/'GUARD_RECEIPT.json').exists(),'No repeat'
def hb(phase): (W/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':case,'phase':phase},indent=2))
hb('prequeue_admission');require_capacity(W/'recurrent_package',d);shutil.copy2(d/'fresh_resource_admission.json',d/'prequeue_resource_admission.json')
argv=['npu-guard','run','--owner','baseline700-20261026/gdn_capacity_recurrent1','--wait','600','--timeout','300','--','bash',str(d/'guard_child.sh')]
hb('guard_wait_or_run');start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
receipt={'argv':argv,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'elapsed_seconds':time.monotonic()-t}
(d/'GUARD_RECEIPT.json').write_text(json.dumps(receipt,indent=2));hb('device_finished_cpu_packaging');print(json.dumps(receipt))
