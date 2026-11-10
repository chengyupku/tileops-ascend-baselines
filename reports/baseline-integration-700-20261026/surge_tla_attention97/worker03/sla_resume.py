from pathlib import Path
from orchestrate import P,R,SSH,OWNER,HOSTS,statuses,remote,beat
import subprocess,json,datetime,time,shlex,tarfile,io,hashlib
TAG='sla_resume_20261027T0529'
keys=json.loads((P/'SLA_REMAINING.json').read_text())
root=P/TAG;root.mkdir(exist_ok=True)
def dump(p,d):p.write_text(json.dumps(d,indent=2))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
plan=dict(utc=now(),task_type='actual_retest_and_interface_integration',keys=keys,owner=OWNER,writes=str(P),reuse=['DEPLOYMENT_RECIPE.json','CPU_ROUTE_CHECK.json','CONTINUATION_CPU_CHECKS.json'],acceptance=['complete original checker all outputs','unchanged original inputs seed tolerance','guard exit0','native source compiler library identity linked'],deadline='2026-10-27T05:31:00+00:00',accepted_added=0)
dump(root/'CONTRACT.json',plan)
checkpoint=False
for key in keys:
 cid=key[2];out=P/'attempts'/(cid+'-'+TAG);out.mkdir(exist_ok=True)
 st=statuses(cid+'-'+TAG+'-pre')
 h=next((h for h in HOSTS[1:] if st[h]==0),HOSTS[1])
 # Reuse verified remote deployment; only deploy unique output-path wrapper and verify dependency identity delta.
 pre=remote(h,'test -f '+R+'/adapter.py && test -f '+R+'/frozen/TileOPs/scripts/ascend_pair_check.py && sha256sum '+R+'/adapter.py '+R+'/capture.py '+R+'/original_sources/examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py && uname -m && head -20 /opt/tileops-all-pairs/bin/allpairs-python',capture_output=True)
 (out/'cpu_preflight.txt').write_bytes(pre.stdout+pre.stderr)
 if pre.returncode:raise RuntimeError('previously verified deployment missing')
 for f in ['adapter.py','capture.py','original_sources/examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py']:
  assert hashlib.sha256((P/f).read_bytes()).hexdigest() in pre.stdout.decode(),f
 remote(h,'cat > '+R+'/sla_resume_case.py',input=(P/'sla_resume_case.py').read_bytes(),check=True)
 cmd=['npu-guard','run','--owner',OWNER,'--wait','600','--timeout','600','--','/opt/tileops-all-pairs/bin/allpairs-python',R+'/sla_resume_case.py',key[1],cid]
 c=dict(key=key,host=h,argv=cmd,requested_at=now(),start=time.time(),changed_condition='Authorized remaining exact key without prior effective execution; original recipe and metadata binding reused',threehost_status=st)
 dump(out/'command.json',c)
 with (out/'guard.log').open('w') as f:
  proc=subprocess.Popen([SSH,h,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   beat(key,'foreground guard request',str(out/'guard.log'),'see guard metadata for queue/execution; original SLA 05:31 unchanged')
   if not checkpoint and time.time()>=1793079060:
    dump(root/'SLA_0531_SNAPSHOT.json',dict(utc=now(),deadline=plan['deadline'],active_key=key,controller_pid=proc.pid,result=json.loads((P/'RESULT.json').read_text()),remaining=keys[keys.index(key):],sla_overdue=True));checkpoint=True
   time.sleep(10)
 c.update(returncode=proc.returncode,end=time.time(),ended_at=now());dump(out/'guard.json',c)
 arc=remote(h,'tar -czf - -C '+R+'/'+TAG+' runs/'+cid+'-pto-A5-original-branch',capture_output=True)
 (out/'remote_evidence.tar.gz').write_bytes(arc.stdout)
 if arc.returncode==0:
  with tarfile.open(fileobj=io.BytesIO(arc.stdout),mode='r:gz') as t:t.extractall(out,filter='data')
 remote_meta=remote(h,'cat /run/lock/ascend-npu/device-0.lock',capture_output=True);(out/'guard_exit_metadata.json').write_bytes(remote_meta.stdout)
 ck=out/'runs'/(cid+'-pto-A5-original-branch')/'checker.json';m=out/'runs'/(cid+'-pto-A5-original-branch')/'GUARD_RUNNING.json'
 checker=json.loads(ck.read_text()) if ck.exists() else {}
 meta=json.loads(m.read_text()) if m.exists() else {}
 success=proc.returncode==0 and checker.get('status')=='verified'
 row=dict(key=key,host=h,status='verified_pending_parent_review' if success else checker.get('status','execution_timeout_no_full_checker' if proc.returncode==124 else 'guard_or_entry_failure'),guard=str(out/'guard.json'),checker=str(ck) if ck.exists() else None,guard_exit=proc.returncode,numerical_success=success,numerical_failure_established=checker.get('stage')=='verify' and checker.get('error_type')=='AssertionError',accepted_added=0,runtime_ready=False,guard_times=dict(requested_at=meta.get('requested_at'),acquired_at=meta.get('acquired_at'),child_start=meta.get('updated_at'),end=c['ended_at']),native_evidence=str(ck.parent),timings=dict(wall_seconds=c['end']-c['start'],queue_seconds=None,physical_idle_wait_seconds=None,held_cpu_seconds=None,device_seconds=None),sla_overdue=time.time()>1793079060)
 dump(out/'RESULT.json',row)
 report=json.loads((P/'RESULT.json').read_text());report['rows']=[row if r['key']==key else r for r in report['rows']];dump(P/'RESULT.json',report)
 print(json.dumps(row),flush=True)
 beat(key,'exact key collected',str(out/'RESULT.json'))
dump(root/'COMPLETED.json',dict(utc=now(),accepted_added=0))
