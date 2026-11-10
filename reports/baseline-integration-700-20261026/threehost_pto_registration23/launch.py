"""Future authorized checker invocation. CPU registration work does not execute this."""
import os,sys,json,subprocess,shlex,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
import registration as M
host,pin=M.current_host();op,cid=sys.argv[1:3]
if (op,cid) not in {(r['op'],r['case_id']) for r in M.rows()}:raise SystemExit('Unknown exact key')
owner='baseline700-20261026/threehost_pto_registration23'
env=dict(os.environ,PTO23_OWNER=owner)
entry=str(P/'runtime_entry.py')
if host=='local':
 command=['bash','-c','source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh && export TORCH_DEVICE_BACKEND_AUTOLOAD=0 FLA_NPU_DISABLE_PTH=1 && exec /usr/bin/python3 -I -S -B '+shlex.join([entry,op,cid])]
else:command=['/opt/tileops-all-pairs/bin/allpairs-python',entry,op,cid]
guard=['npu-guard','run','--owner',owner,'--wait','600','--timeout','300','--',*command]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
r=subprocess.run(guard,env=env)
receipt=dict(key=['tilelang-ascend',op,cid],host=host,argv=guard,requested=start,acquired=None,child_start=None,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),guard_exit=r.returncode,queue_seconds=None,physical_idle_wait_seconds=None,held_cpu_seconds=None,device_seconds=None)
(P/('launch_'+op+'_'+cid+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(r.returncode)
