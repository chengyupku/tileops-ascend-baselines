import subprocess,json,time,sys,datetime,os
from pathlib import Path
P=Path(__file__).resolve().parent
case=sys.argv[1]; d=P/case
assert case in ('throughput-4k-complex128','scientific-1m-complex128')
assert not (d/'GUARD_RESULT.json').exists()
cmd=['npu-guard','run','--owner','baseline700-20261026/fft_remaining_c128_2','--wait','1200','--timeout','180','--','.tools/baselines/baseline-run','ops-math','python',str(P/'runner.py'),case]
(d/'COMMAND.json').write_text(json.dumps(cmd,indent=2))
for label in ['BEFORE']:
 r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);(d/f'STATUS_{label}.json').write_text(r.stdout)
(P/'HEARTBEAT.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'guard_queue_or_run','case':case}))
begin=time.monotonic()
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
with (d/'guard.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
(d/'GUARD_RESULT.json').write_text(json.dumps({'returncode':r.returncode,'wall_seconds':time.monotonic()-begin},indent=2))
r2=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);(d/'STATUS_AFTER.json').write_text(r2.stdout)
print((d/'GUARD_RESULT.json').read_text());print((d/'checker.json').read_text() if (d/'checker.json').exists() else (d/'guard.log').read_text()[-5000:])
