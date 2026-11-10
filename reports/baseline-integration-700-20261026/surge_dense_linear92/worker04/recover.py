import batch,json,time,datetime,subprocess,concurrent.futures
W=batch.W
cases=['q35-gva-decode-b128-float16','q35-gva-decode-b256-float16','q35-397b-tp1-32x8192-float16']
for i,case in enumerate(cases,8):
 a=f'attempt{i:02d}'
 while True:
  batch.heartbeat(case,'waiting outside guard for compatible host state change')
  s=batch.status(batch.HOST)
  (W/'RECOVERY_HOST_STATUS.json').write_text(json.dumps(s,indent=2))
  try:state=json.loads(s['stdout'])
  except ValueError:state={}
  with (W/'RECOVERY_POLL.jsonl').open('a') as f:f.write(json.dumps(dict(utc=batch.now(),state=state.get('state'),owner=state.get('reservation',{}).get('metadata',{}).get('owner'),pids=state.get('pids'),errors=state.get('errors')))+'\n')
  if state.get('state') in ('idle','busy') and not state.get('proc_fd_errors') and not any(p.get('pid') in (890863,892989,894777) for p in state.get('pids',[])):break
  if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime(2026,10,27,5,15,tzinfo=datetime.timezone.utc):raise SystemExit('no compatible host slot before deadline')
  time.sleep(30)
 with concurrent.futures.ThreadPoolExecutor() as e:states=list(e.map(batch.status,['local','openlibing-DevEnv-661495',batch.HOST]))
 (W/(a+'_THREEHOST.json')).write_text(json.dumps(states,indent=2))
 (W/(a+'_CONDITION_CHANGE.json')).write_text(json.dumps(dict(case=case,utc=batch.now(),prior='guard75 queue timeout or guard78 transient unreadable /proc',now='server2 visibility restored and previous blocking process exited; current cooperative FIFO may still be busy; renewed bounded reservation',evidence=str(W/'RECOVERY_HOST_STATUS.json')),indent=2))
 with (W/(a+'_transport.log')).open('w') as log:
  p=subprocess.Popen([batch.SSH,batch.HOST,f'cd {batch.R} && /opt/tileops-ci/runtime/bin/python remote_run.py {case} {a}'],stdout=log,stderr=subprocess.STDOUT)
  while True:
   try:p.wait(timeout=30);break
   except subprocess.TimeoutExpired:batch.heartbeat(case,'retry after documented host-state change; foreground guard')
 batch.collect(a);batch.result(case,a)
