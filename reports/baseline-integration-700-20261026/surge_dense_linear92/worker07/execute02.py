import pathlib,subprocess,json,shlex,datetime,time
P=pathlib.Path(__file__).resolve().parent;rel=P.relative_to('/root/dyq/workspace');R='/opt/tileops-all-pairs/workspace/'+str(rel)
SSH='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh';H='openlibing-DevEnv-055409'
def remote(cmd,**kw):return subprocess.run([SSH,H,cmd],**kw)
files=['SCOPE.json','run_case.py','baseline-run','native_log_capture.py','integration700_sgl_gdn_remote38.py','route_install.py','CHECKER_PIN.json','guard_one02.py']
tar=subprocess.run(['tar','-C',str(P),'-cf','-',*files],capture_output=True,check=True).stdout
remote('mkdir -p '+shlex.quote(R)+' && tar -C '+shlex.quote(R)+' -xf -',input=tar,check=True)
rows=json.loads((P/'SCOPE.json').read_text())[4:6]
for i,row in enumerate(rows):
 cid=row['case_id']; d=P/('retry'+str(i+4));d.mkdir(exist_ok=True)
 (P/'HEARTBEAT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'key':row['key'],'phase':'three-host status then exact checker','last_output':str(d),'next':'collect complete receipt'}))
 for host in ['local','openlibing-DevEnv-661495',H]:
  cmd=['npu-guard','status','--json'] if host=='local' else [SSH,host,'npu-guard status --json']
  result=subprocess.run(cmd,capture_output=True,text=True);(d/(host+'.status.json')).write_text(json.dumps({'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr}))
 with (d/'submit.log').open('w') as f: result=remote('python3 '+shlex.quote(R+'/guard_one02.py')+' '+shlex.quote(cid),stdout=f,stderr=subprocess.STDOUT)
 artifact='runs/exact02--GDNFwdOp--'+cid
 data=remote('tar -C '+shlex.quote(R)+' -cf - '+shlex.quote(artifact),capture_output=True,check=True).stdout
 subprocess.run(['tar','-C',str(P),'-xf','-'],input=data,check=True)
 run=P/artifact;checker=run/'checker.json'; guard=json.loads((run/'GUARD_RESULT.json').read_text()); c=json.loads(checker.read_text()) if checker.exists() else {}
 rr={'key':row['key'],'status':c.get('status','not_completed'),'reason':c.get('error',c.get('reason')),'checker':str(checker) if checker.exists() else None,'guard':str(run/'GUARD_RESULT.json'),'guard_exit':guard['exit_code'],'registered':False,'accepted_added':0,'executed_this_task':guard['metadata'] is not None,'native':str(run/'output/native.jsonl'),'runtime_ready':False}
 (d/'RESULT.json').write_text(json.dumps(rr,indent=2));print(json.dumps(rr),flush=True)
