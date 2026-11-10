from pathlib import Path
import subprocess,json,time,datetime,shlex,sys,tarfile,io,shutil,os
P=Path(__file__).resolve().parent
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
HOSTS=['local','openlibing-DevEnv-661495','openlibing-DevEnv-055409'];OWNER='baseline700-20261026/surge_tla_attention97/worker04'
R='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/surge_tla_attention97/worker04';OLD='/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/threehost_attention_mla_mha31'
def remote(h,c,**kw):return subprocess.run([SSH,h,c],**kw)
def beat(key,phase,output,block=None):
 tmp=P/('HEARTBEAT.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_key=key,phase=phase,last_output=output,blocker=block,next='complete original checker and preserve evidence'),indent=2));tmp.replace(P/'HEARTBEAT.json')
def statuses(tag):
 from concurrent.futures import ThreadPoolExecutor
 def one(h):
  c=['npu-guard','status','--json'] if h=='local' else [SSH,h,'npu-guard status --json']
  r=subprocess.run(c,capture_output=True,text=True);(P/(tag+'-'+h+'-'+str(time.time_ns())+'.json')).write_text(r.stdout);return h,r.returncode
 with ThreadPoolExecutor(3) as pool:return dict(pool.map(one,HOSTS))
def deploy(h):
 remote(h,'mkdir -p '+R,check=True)
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode='w') as t:
  for f in ['adapter_direct.py','capture.py','continuation_run_direct.py','SCOPE.json']:t.add(P/f,arcname=f)
 remote(h,'tar -xf - -C '+R,input=data.getvalue(),check=True)
 remote(h,'ln -sfn '+OLD+'/frozen '+R+'/frozen; ln -sfn '+OLD+'/private_tla '+R+'/private_tla; test -f /opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_tla_compiler/handoff/lib/libtilelang.so; test -f /opt/tileops-all-pairs/workspace/ascend_baselines/tilelang-ascend/3rdparty/pto-isa/include/pto/pto-inst.hpp',check=True)
def run(cid):
 scope=json.loads((P/'SCOPE.json').read_text());row=next(r for r in scope if r['case_id']==cid);key=row['key'];op=row['operator']
 st=statuses(cid+'-pre');h=next((x for x in HOSTS[2:]+HOSTS[1:2]+HOSTS[:1] if st[x]==0),'openlibing-DevEnv-055409')
 if all(st.get(x)!=0 for x in HOSTS):
  prior=[json.loads(f.read_text()) for f in (P/'continuation_20261027T0440'/'attempts').glob('*/guard.json')]
  h=min(HOSTS[2:]+HOSTS[1:2]+HOSTS[:1],key=lambda host:sum(g['returncode']==75 and g['host']==host for g in prior))
 active_hosts={json.loads(f.read_text())['host'] for f in (P/'continuation_20261027T0440'/'attempts').glob('*/command.json') if not (f.parent/'guard.json').exists()}
 available=[x for x in HOSTS if x not in active_hosts]
 if h in active_hosts and available:h=next((x for x in available if st.get(x)==0),available[0])
 forced=sys.argv[2] if len(sys.argv)>2 else None
 if forced:h=forced
 tag=('-'+sys.argv[3]) if len(sys.argv)>3 else ''
 out=P/'continuation_20261027T0440'/'attempts'/(cid+'-original-direct'+('-'+h if forced else '')+tag);out.mkdir(parents=True,exist_ok=True)
 assert not (out/'guard.json').exists(), 'No unchanged automatic rerun'
 if h!='local':deploy(h)
 cmd=['npu-guard','run','--owner',OWNER,'--wait','600','--timeout','300','--']
 cmd+= [str(P/'local-python'),str(P/'continuation_run_direct_local.py'),op,cid] if h=='local' else ['/opt/tileops-all-pairs/bin/allpairs-python',R+'/continuation_run_direct.py',op,cid]
 r=dict(key=key,host=h,argv=cmd,requested_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),start=time.time(),changed_condition=(sys.argv[4] if len(sys.argv)>4 else ('Migrate after guard visibility failure / no checker; different compatible host' if forced else 'Changed original API: original flash_attention_fwd with D128, Sq1 complete-window layout binding; expert route was infrastructure-aborted. No kernel/pass modification.')))
 (out/'command.json').write_text(json.dumps(r,indent=2));beat(key,'guard run',str(out/'command.json'))
 with (out/'guard.log').open('w') as f:
  proc=subprocess.Popen(cmd if h=='local' else [SSH,h,shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   time.sleep(20);beat(key,'guard run',str(out/'guard.log'))
 r.update(returncode=proc.returncode,end=time.time(),ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'guard.json').write_text(json.dumps(r,indent=2))
 if h!='local':
  arc=remote(h,'tar --exclude=inputs.pt -czf - -C '+R+' continuation_20261027T0440/runs/'+cid+'-pto-A5-original-direct',capture_output=True)
  (out/'remote_evidence.tar.gz').write_bytes(arc.stdout)
  if arc.returncode==0:
   with tarfile.open(fileobj=io.BytesIO(arc.stdout),mode='r:gz') as t:t.extractall(out,filter='data')
 if h=='local':
  source=P/'continuation_20261027T0440'/'local_runs'/(cid+'-pto-A5-original-direct')
  if source.exists():shutil.copytree(source,out/'local_evidence',dirs_exist_ok=True,ignore=shutil.ignore_patterns('inputs.pt'))
 checker=list(out.rglob('checker.json')) if h!='local' else list((P/'continuation_20261027T0440'/'local_runs'/ (cid+'-pto-A5-original-direct')).glob('checker.json'))
 result=dict(key=key,status='attempt_complete_not_verified',guard=str(out/'guard.json'),checker=[str(f) for f in checker],numerical_success=False,numerical_failure_established=False,accepted_added=0,host=h)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));beat(key,'attempt complete',str(out/'RESULT.json'))
 print(json.dumps(r),flush=True)
if __name__=='__main__':
 assert sys.argv[1]=='qwen3-30b-4k-float16'
 run(sys.argv[1])
