import pathlib,json,subprocess,time,hashlib,shlex
P=pathlib.Path('/root/dyq/workspace/reports/baseline-flex-20261106');B=P/'batches/batch22';rows=json.load(open(B/'PLAN.json'))['rows']
for h,host in [(2,'openlibing-DevEnv-055409'),(1,'openlibing-DevEnv-661495')]:
 d=P/f'server{h}'
 while True:
  s=json.load(open(d/'STATE.json'))
  if s['phase']=='stopped' and not pathlib.Path('/proc/'+str(s['pid'])).exists():break
  time.sleep(3)
 print('DRAINED',h,flush=True)
 f=d/'measure_pair.py';before=f.read_bytes();(B/f'server{h}-measure.before.py').write_bytes(before)
 src=before.decode();needle="(op == 'GDNFwdOp' and library == 'vllm-ascend')";assert needle in src
 src=src.replace(needle,"(op in {'GDNFwdOp','DeltaNetFwdOp'} and library == 'vllm-ascend')");compile(src,str(f),'exec');f.write_text(src)
 remote=f'/opt/baseline-flex-20261106/server{h}/measure_pair.py'
 subprocess.run(['scp','-O','-q','-o','ClearAllForwardings=yes',str(f),host+':'+remote+'.next'],check=True)
 code='import pathlib,hashlib;p=pathlib.Path('+repr(remote+'.next')+');assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(f.read_bytes()).hexdigest())+';p.replace('+repr(remote)+')'
 subprocess.run(['ssh','-o','ClearAllForwardings=yes',host,'python3 -c '+shlex.quote(code)],check=True)
 plan=d/'PLAN.json';old=plan.read_bytes();(B/f'server{h}-PLAN.before.json').write_bytes(old);v=json.loads(old);new=[t for t in rows if t['host']==f'server{h}'];assert not {t['id'] for t in new}&{t['id'] for t in v['rows']};v['rows']=new+v['rows'];tmp=plan.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(plan)
 (d/'STOP').unlink()
 subprocess.run(['systemd-run',f'--unit=baseline-flex-server{h}','--property=WorkingDirectory=/root/dyq/workspace','/usr/bin/python3',str(d/'controller.py')],check=True)
 print('RESTARTED_WITH_NEW_QUEUE',h,len(new),flush=True)
(B/'DEPLOYED.json').write_text(json.dumps({'at':time.time(),'added':len(rows),'safe_stop_both':True},indent=2))
