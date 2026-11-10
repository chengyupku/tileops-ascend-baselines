from pathlib import Path
from orchestrate import P,R,SSH,OWNER,HOSTS,statuses,remote
import subprocess,json,datetime,time,shlex,tarfile,io
TAG='continuation_20261027T0508'
CID='ds-v2-4k-bfloat16'
KEY=['tilelang-ascend','MLADecodeWithKVCacheFwdOp',CID]
h='openlibing-DevEnv-661495'
out=P/'attempts'/(CID+'-'+TAG);out.mkdir(exist_ok=True)
assert time.time()+1230 < datetime.datetime(2026,10,27,5,31,tzinfo=datetime.timezone.utc).timestamp()
st=statuses(TAG+'-presubmit')
# Already deployed original libraries, source, adapter and checker reused unchanged.
remote(h,'test -f '+R+'/adapter.py && test -f '+R+'/original_sources/examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py && test -e '+R+'/frozen/TileOPs/scripts/ascend_pair_check.py',check=True)
remote(h,'cat > '+R+'/'+TAG+'_case.py',input=(P/(TAG+'_case.py')).read_bytes(),check=True)
cmd=['npu-guard','run','--owner',OWNER,'--wait','600','--timeout','600','--','/opt/tileops-all-pairs/bin/allpairs-python',R+'/'+TAG+'_case.py',KEY[1],CID]
c={'key':KEY,'host':h,'argv':cmd,'requested_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start':time.time(),'changed_condition':'Never executed exact key; H128 REPLICATE_H=2 differs from previously attempted H64 branch','threehost_status':st}
(out/'command.json').write_text(json.dumps(c,indent=2))
with (out/'guard.log').open('w') as f:
 proc=subprocess.Popen([SSH,h,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 while proc.poll() is None:
  heartbeat={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_key':KEY,'phase':'independent H128 foreground guard request','last_output':str(out/'guard.log'),'blocker':'guard queue or original device execution; consult guard log','next':'collect complete original checker','controller_pid':proc.pid}
  (P/(TAG+'_HEARTBEAT.json')).write_text(json.dumps(heartbeat,indent=2))
  # Previous controller retains root heartbeat until its request completes.
  if not Path('/proc/3365781').exists():(P/'HEARTBEAT.json').write_text(json.dumps(heartbeat,indent=2))
  time.sleep(20)
c.update(returncode=proc.returncode,end=time.time(),ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(c,indent=2))
a=remote(h,'tar -czf - -C '+R+'/'+TAG+' runs/'+CID+'-pto-A5-original-branch',capture_output=True)
(out/'remote_evidence.tar.gz').write_bytes(a.stdout)
if a.returncode==0:
 with tarfile.open(fileobj=io.BytesIO(a.stdout),mode='r:gz') as t:t.extractall(out,filter='data')
ck=out/'runs'/(CID+'-pto-A5-original-branch')/'checker.json'
result={'key':KEY,'guard_exit':proc.returncode,'guard':str(out/'guard.json'),'checker':str(ck) if ck.exists() else None,'checker_status':json.loads(ck.read_text()).get('status') if ck.exists() else None,'accepted_added':0,'runtime_ready':False}
result['numerical_success']=result['guard_exit']==0 and result['checker_status']=='verified'
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
