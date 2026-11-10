from pathlib import Path
import subprocess,json,time,sys
P=Path(__file__).resolve().parent
if (P/'STOP_REMOTE').exists():(P/'STOP_REMOTE').rename(P/'STOP_REMOTE.completed')
rows=json.loads((P/'SCOPE.json').read_text());keys=[(r['baseline'],r['operator'],r['case_id']) for r in rows];keys.sort(key=lambda k:(0 if '128' in k[2] else 1 if '256' in k[2] else 2 if '512' in k[2] else 3,k))
active={};finished=set();blocked_signatures={};attempts={};no_progress=time.time()
def runscript(script,args):return subprocess.run([sys.executable,str(P/script),*args],check=True)
def status(host):
 cmd=['npu-guard','status','--json'] if host=='local' else [sys.executable,str(P/'transport.py'),host,'npu-guard status --json']
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 try:return json.loads(r.stdout)
 except Exception:return dict(state='unknown',error=r.stdout+r.stderr)
while True:
 for host,a in list(active.items()):
  rc=a['proc'].poll()
  if rc is None:continue
  key=a['key'];g=json.loads((P/(a['tag']+'.guard.json')).read_text());a['stream'].close()
  if host!='local':runscript('pull.py',[host])
  runscript('checkpoint.py',[])
  if g['returncode'] in (75,76,78):blocked_signatures[(key,host)]=a['signature']
  else:finished.add(key);no_progress=time.time()
  del active[host]
 result=json.loads((P/'RESULT.json').read_text());verified={tuple(r['key']) for r in result['rows'] if r['status']=='full_checker_native_verified'}
 pending=[k for k in keys if k not in verified and k not in finished and k not in [a['key'] for a in active.values()]]
 if not pending and not active:break
 snapshots={}
 for host in ['local','server1','server2']:
  if host in active:continue
  if host=='local' and not (P/'tilelang-ascend_BmmFwdOp_square-256-bfloat16_local_v2.guard.json').exists():continue
  s=status(host);snapshots[host]=s;res=s.get('reservation',{});meta=res.get('metadata') or {};sig=[s.get('state'),meta.get('pid'),meta.get('start_ticks'),meta.get('phase'),res.get('locked'),[(x.get('pid'),x.get('start_ticks')) for x in s.get('pids',[])]]
  cooperative=res.get('locked') and meta.get('owner')!='baseline700-20261026/threehost_pto_original23'
  if s.get('state')!='idle' and not cooperative:continue
  candidates=[k for k in pending if blocked_signatures.get((k,host))!=sig]
  if not candidates:continue
  key=candidates[0];pending.remove(key);attempts[key]=attempts.get(key,0)+1;suffix='_route'+(sys.argv[1] if len(sys.argv)>1 else '')+str(attempts[key])+'_'+host;tag='_'.join(key)+suffix
  (P/(tag+'.routing.json')).write_text(json.dumps(s,indent=2))
  argv=[sys.executable,str(P/('launch_local.py' if host=='local' else 'launch.py')),*(list(key[1:]) if host=='local' else [host,*key]),suffix]
  stream=(P/(tag+'.controller.log')).open('w');proc=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT);active[host]=dict(proc=proc,key=key,tag=tag,signature=sig,stream=stream)
 (P/'ROUTING_HEARTBEAT.json').write_text(json.dumps(dict(at=time.time(),phase='threehost_guarded_execution',verified=len(verified),active={h:dict(key=a['key'],tag=a['tag'],controller_pid=a['proc'].pid) for h,a in active.items()},pending=len(pending),host_status=snapshots),indent=2))
 if not active and time.time()-no_progress>900:break
 time.sleep(10)
runscript('checkpoint.py',[]);print('ROUTED BATCH COMPLETE')
