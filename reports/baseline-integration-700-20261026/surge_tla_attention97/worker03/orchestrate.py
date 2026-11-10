from pathlib import Path
import subprocess,json,time,datetime,shlex,sys,tarfile,io,os
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409'];OWNER='baseline700-20261026/surge_tla_attention97/worker03'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker03';OLD='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_mla_mha31'
def remote(h,c,**kw):return subprocess.run([SSH,h,c],**kw)
def beat(key,phase,output,block=None):
 (P/'HEARTBEAT.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_key=key,phase=phase,last_output=output,blocker=block,next='complete original checker and preserve evidence'),indent=2))
def statuses(tag):
 from concurrent.futures import ThreadPoolExecutor
 def one(h):
  c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
  r=subprocess.run(c,capture_output=True,text=True);(P/(tag+'-'+h+'.json')).write_text(r.stdout);return h,r.returncode
 with ThreadPoolExecutor(3) as pool:return dict(pool.map(one,HOSTS))
def deploy(h):
 remote(h,'mkdir -p '+R,check=True)
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode='w') as t:
  for f in ['adapter.py','capture.py','run_case.py','SCOPE.json','original_sources','cpu_preflight.py']:t.add(P/f,arcname=f)
 remote(h,'tar -xf - -C '+R,input=data.getvalue(),check=True)
 remote(h,'ln -sfn '+OLD+'/frozen '+R+'/frozen; ln -sfn '+OLD+'/private_tla '+R+'/private_tla; test -f /opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_tla_compiler/handoff/lib/libtilelang.so; test -f /opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend/3rdparty/pto-isa/include/pto/pto-inst.hpp',check=True)
def run(cid):
 scope=json.loads((P/'SCOPE.json').read_text());row=next(r for r in scope if r['case_id']==cid);key=row['key'];op=row['operator']
 label=os.environ.get('ATTEMPT_LABEL','');st=statuses(cid+label+'-pre');h=next((x for x in HOSTS[1:]+HOSTS[:1] if st[x]==0),os.environ.get('HOST_PREFERENCE','local'))
 out=P/'attempts'/(cid+label);out.mkdir(parents=True,exist_ok=True)
 if h!='local':deploy(h)
 if label=='-libpathfix01':
  # Preserve previous run directories before a permitted retry; never collect stale receipts.
  dirname=cid+'-pto-A5-original-branch'
  if h!='local':
   src=R+'/runs/'+dirname;dst=R+'/previous_runs/'+dirname+label
   remote(h,'mkdir -p '+shlex.quote(R+'/previous_runs')+'; if test -d '+shlex.quote(src)+'; then test ! -e '+shlex.quote(dst)+' && mv '+shlex.quote(src)+' '+shlex.quote(dst)+'; fi',check=True)
  else:
   src=P/'local_runs'/dirname;dst=P/'previous_local_runs'/(dirname+label)
   if src.exists():dst.parent.mkdir(exist_ok=True);src.rename(dst)
 cmd=['npu-guard','run','--owner',OWNER,'--wait','600','--timeout','600','--']
 cmd+= [str(P/'local-python'),str(P/'run_local.py'),op,cid] if h=='local' else ['/opt/tileops-all-pairs/bin/allpairs-python',R+'/run_case.py',op,cid]
 r=dict(key=key,host=h,argv=cmd,requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),start=time.time(),changed_condition=os.environ.get('CHANGED_CONDITION','Previously unrun exact case; no sibling failure substituted'))
 (out/'command.json').write_text(json.dumps(r,indent=2));beat(key,'guard run',str(out/'command.json'))
 with (out/'guard.log').open('w') as f:
  proc=subprocess.Popen(cmd if h=='local' else [SSH,h,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   time.sleep(20);beat(key,'guard run',str(out/'guard.log'))
 r.update(returncode=proc.returncode,end=time.time(),ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(r,indent=2))
 if h!='local':
  arc=remote(h,'tar -czf - -C '+R+' runs/'+cid+'-pto-A5-original-branch',capture_output=True)
  (out/'remote_evidence.tar.gz').write_bytes(arc.stdout)
  if arc.returncode==0:
   with tarfile.open(fileobj=io.BytesIO(arc.stdout),mode='r:gz') as t:t.extractall(out,filter='data')
 checker=list(out.rglob('checker.json')) if h!='local' else list((P/'local_runs'/ (cid+'-pto-A5-original-branch')).glob('checker.json'))
 result=dict(key=key,status='attempt_complete_not_verified',guard=str(out/'guard.json'),checker=[str(f) for f in checker],numerical_success=False,numerical_failure_established=False,accepted_added=0,host=h)
 if checker:
  c=json.loads(checker[0].read_text());result['checker_status']=c.get('status');result['numerical_success']=c.get('status')=='verified' and r['returncode']==0
  result['numerical_failure_established']=c.get('status')=='failed' and c.get('stage')=='verify' and c.get('error_type')=='AssertionError'
  result['status']='verified_pending_parent_review' if result['numerical_success'] else c.get('status','unknown')
 elif r['returncode']==124:result['status']='execution_timeout_no_full_checker'
 elif r['returncode'] in [75,78]:result['status']='guard_refused_or_visibility_lost'
 elif r['returncode']==76:result['status']='guard_external_interference_invalid'
 result['runtime_ready']=False
 result['timings']={'controller_wall_seconds':r['end']-r['start'],'queue_seconds':None,'physical_idle_wait_seconds':None,'held_cpu_seconds':None,'device_seconds':None}
 metas=list(out.rglob('GUARD_RUNNING.json')) if h!='local' else list((P/'local_runs'/(cid+'-pto-A5-original-branch')).glob('GUARD_RUNNING.json'))
 if metas:
  m=json.loads(metas[0].read_text());result['guard_times']={x:m.get(x) for x in ['requested_at','acquired_at','updated_at']};result['child']=m.get('child')
 (out/'RESULT.json').write_text(json.dumps(result,indent=2))
 report=json.loads((P/'RESULT.json').read_text())
 report['rows']=[result if x['key']==key else x for x in report['rows']]
 (P/'RESULT.json').write_text(json.dumps(report,indent=2));beat(key,'attempt complete',str(out/'RESULT.json'))
 print(json.dumps(r),flush=True)
if __name__=='__main__':run(sys.argv[1])
