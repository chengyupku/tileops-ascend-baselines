import subprocess,pathlib,datetime,json,time,os
W=pathlib.Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/linear_deployment_recovery');d=W/'decode-b1-float16'
f=W/'FEEDBACK.txt'
if f.exists(): print(f.read_text(),flush=True)
a=['npu-guard','run','--owner','baseline700-20261026/linear_deployment_recovery','--wait','3600','--timeout','300','--','bash',str(d/'child.sh')]
s=time.time();r=dict(argv=a,requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
with (d/'guard.stdout.log').open('w') as o,(d/'guard.stderr.log').open('w') as e: p=subprocess.run(a,stdout=o,stderr=e)
r.update(returncode=p.returncode,elapsed_seconds=time.time()-s,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(d/'GUARD_RECEIPT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
