import json,subprocess,datetime,sys,time,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(W/'chunk_package'));from resource_gate import require_capacity
case=sys.argv[1];d=W/case
scope=json.loads((W/'SCOPE.json').read_text());assert case in [x['case_id'] for x in scope]
first='q35-397b-tp1-32768-bfloat16'
if case!=first:
 assert json.loads((W/first/'checker.json').read_text())['status']=='verified','firstcase must verify'
assert not (d/'GUARD_RECEIPT.json').exists(),'Never retry same attempt'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def hb(phase):
 (W/'HEARTBEAT.json').write_text(json.dumps({'utc':now(),'current_case':case,'phase':phase,'accepted_added':0},indent=2))
hb('prequeue_capacity')
try:require_capacity(W/'chunk_package',d)
except Exception as e:
 (d/'PREQUEUE_REFUSAL.json').write_text(json.dumps({'utc':now(),'error':str(e)},indent=2));hb('capacity_refused');raise
shutil.copyfile(d/'fresh_resource_admission.json',d/'prequeue_resource_admission.json')
# Snapshot actual running guard before invoking the original child. No device imports.
inner="import json,os,sys,datetime; from pathlib import Path; d=Path(sys.argv[1]); (d/'GUARD_RUNNING.json').write_text(Path('/run/lock/ascend-npu/device-0.lock').read_text()); (d/'CHILD_STARTED.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})); os.execvp('bash',['bash',str(d/'child.sh')])"
argv=['npu-guard','run','--owner','baseline700-20261026/gdn_capacity_chunk16','--wait','600','--timeout','300','--','python3','-S','-B','-c',inner,str(d)]
t=time.time();start=now();hb('guard_wait_or_run')
with (d/'guard.stdout.log').open('w') as out,(d/'guard.stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err)
end=now();elapsed=time.time()-t
q=None
if (d/'CHILD_STARTED.json').exists():q=(datetime.datetime.fromisoformat(json.loads((d/'CHILD_STARTED.json').read_text())['utc'])-datetime.datetime.fromisoformat(start)).total_seconds()
receipt={'argv':argv,'start':start,'end':end,'returncode':p.returncode,'elapsed_seconds':elapsed,'queue_seconds':q,'child_seconds':None if q is None else elapsed-q}
(d/'GUARD_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
hb('guard_released')
x=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
print(json.dumps({'case':case,'guard_rc':p.returncode,'status':x.get('status'),'stage':x.get('stage'),'error':x.get('error'),'seconds':elapsed,'queue_seconds':q}),flush=True)
