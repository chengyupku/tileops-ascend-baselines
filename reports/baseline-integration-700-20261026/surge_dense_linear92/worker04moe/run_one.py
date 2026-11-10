import pathlib,json,sys,subprocess,os,datetime,time
T=pathlib.Path(__file__).resolve().parent;case=sys.argv[1];P=T/case
owner='baseline700-20261026/surge_dense_linear92/worker04moe'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
cmd=['npu-guard','run','--owner',owner,'--wait','600','--timeout','600','--','bash',str(T/'environment.sh'),'/root/dyq/workspace/.venv/bin/python','-B',str(T/'capture.py'),str(P/'result'),'MoEGroupedGemmFwdOp',case]
env=dict(os.environ,BF16_GEMM_GUARD_OWNER=owner,BF16_GEMM_RUN_ROOT=str(P/'rt'))
request=now();(P/'COMMAND.json').write_text(json.dumps({'requested':request,'command':cmd,'env':{k:env[k] for k in ['BF16_GEMM_GUARD_OWNER','BF16_GEMM_RUN_ROOT']}},indent=2))
(T/'HEARTBEAT.json').write_text(json.dumps({'utc':now(),'ownership':'exclusive SCOPE accepted','key':['ops-transformer','MoEGroupedGemmFwdOp',case],'phase':'guard submitted wait600 timeout600','last_output':'CPU_CHECKS.json; original successful down recipe reused','next':'full unchanged canonical checker','accepted_added':0},indent=2))
start=time.monotonic()
with (P/'guard.stdout.log').open('w') as o,(P/'guard.stderr.log').open('w') as e:
 proc=subprocess.Popen(cmd,env=env,stdout=o,stderr=e)
 while True:
  try:rc=proc.wait(timeout=45);break
  except subprocess.TimeoutExpired:
   phase=P/'result/PHASE.json'
   (T/'HEARTBEAT.json').write_text(json.dumps({'utc':now(),'ownership':'exclusive SCOPE accepted','key':['ops-transformer','MoEGroupedGemmFwdOp',case],'phase':'guard foreground wait/run','last_output':json.loads(phase.read_text()) if phase.exists() else 'queued','guard_pid':proc.pid,'elapsed_s':time.monotonic()-start,'accepted_added':0},indent=2))
end=now();guard={'requested':request,'end':end,'exit_code':rc,'wall_seconds':time.monotonic()-start,'command':cmd}
try:guard['metadata_after']=json.loads(pathlib.Path('/run/lock/ascend-npu/device-0.lock').read_text())
except Exception:pass
(P/'GUARD_RESULT.json').write_text(json.dumps(guard,indent=2))
receipt=P/'result/representative.json'; checker=json.loads(receipt.read_text()) if receipt.exists() else None
row=dict(key=['ops-transformer','MoEGroupedGemmFwdOp',case],task_type='actual integration supplementary execution',status=checker.get('status') if checker else 'checker_not_completed',checker=str(receipt) if checker else None,guard=str(P/'GUARD_RESULT.json'),guard_exit=rc,new_device_execution=(P/'result/guard_snapshot.json').exists(),native=str(P/'result/NATIVE_CALLS.json'),accepted_added=0,registered=False,runtime_ready=False,timings=guard,checker_receipt=checker,prior_scope_path=next(x['prior_scope_path'] for x in json.loads((T/'SCOPE.json').read_text()) if x['key'][2]==case))
(P/'RESULT.json').write_text(json.dumps(row,indent=2));results=[json.loads(p.read_text()) for p in sorted(T.glob('ds-v3-*/RESULT.json'))];(T/'RESULT.json').write_text(json.dumps(results,indent=2));print(json.dumps({'case':case,'guard_exit':rc,'status':row['status'],'seconds':guard['wall_seconds']}))
