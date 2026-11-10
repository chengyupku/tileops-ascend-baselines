import json,time,subprocess,datetime,sys
from pathlib import Path
W=Path(__file__).resolve().parent
stage,limit,script=sys.argv[1:]
out=W/stage;out.mkdir(exist_ok=True)
if (out/'GUARD_RECEIPT.json').exists():raise RuntimeError('No unchanged retry')
if stage.startswith('q35-'):
 sys.dont_write_bytecode=True;sys.path.insert(0,str(W/'chunk_package'))
 from resource_gate import require_capacity
 admission=require_capacity(W/'chunk_package',out)
 (out/'prequeue_resource_admission.json').write_text(json.dumps(admission,indent=2))
argv=['npu-guard','run','--owner','baseline700-20261026/bf16_bmm_conv_grouped_native10','--wait','600','--timeout',limit,'--','bash',script]
t=time.time();start=datetime.datetime.now(datetime.timezone.utc).isoformat();samples=[]
with (out/'guard.stdout.log').open('w') as stdout,(out/'guard.stderr.log').open('w') as stderr:
 p=subprocess.Popen(argv,stdout=stdout,stderr=stderr)
 while p.poll() is None:
  now=datetime.datetime.now(datetime.timezone.utc).isoformat()
  try:
   m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
   if m.get('owner')==argv[3] and m.get('pid')==p.pid:
    if not samples or samples[-1].get('phase')!=m.get('phase'):samples.append(m)
  except Exception:pass
  (W/'HEARTBEAT.json').write_text(json.dumps({'at':now,'stage':stage,'phase':'guard_wait_or_run','elapsed':time.time()-t,'accepted_added':0}))
  time.sleep(5)
result={'argv':argv,'start':start,'end':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'elapsed_seconds':time.time()-t,'guard_samples':samples}
(out/'GUARD_RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);sys.exit(p.returncode)
