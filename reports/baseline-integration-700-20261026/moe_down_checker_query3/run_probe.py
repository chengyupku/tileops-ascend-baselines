import json,os,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
id=sys.argv[1];assert id in [s['id'] for s in json.loads((H/'PLAN.json').read_text())]
attempt=sys.argv[2] if len(sys.argv)>2 else ""
R=H/(id+attempt);R.mkdir(exist_ok=False)
env=dict(os.environ,BF16_GEMM_GUARD_OWNER='baseline700-20261026/moe_down_checker_query3',BF16_GEMM_RUN_ROOT=str(R/'runtime'))
cmd=['npu-guard','run','--owner',env['BF16_GEMM_GUARD_OWNER'],'--wait','600','--timeout','120','--','bash',str(H.parent/'fp16_local_remaining3/environment.sh'),'/usr/bin/python3',str(H/'probe.py'),id,'--device-query','--attempt',attempt]
(R/'command.json').write_text(json.dumps(cmd,indent=2));start=time.time()
with (R/'stdout.log').open('w') as out,(R/'stderr.log').open('w') as err:
 rc=subprocess.call(cmd,env=env,stdout=out,stderr=err)
receipt={'command':cmd,'start':start,'end':time.time(),'rc':rc}
try:receipt['guard_metadata_after']=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
except Exception as e:receipt['metadata_error']=repr(e)
(R/'guard_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));sys.exit(rc)
