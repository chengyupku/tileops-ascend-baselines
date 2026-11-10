import subprocess,json,pathlib,shlex,time,sys
P=pathlib.Path(__file__).resolve().parent
R='/opt/tileops-baseline-cache/workspace';E=R+'/reports/baseline-v3-sample20-20261101/server1'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ClearAllForwardings=yes','openlibing-DevEnv-661495']
rows=json.loads((P/'SAMPLE.json').read_text())['rows']
indices=[int(x) for x in sys.argv[1:]] or list(range(20))
for i in indices:
 r=rows[i];tid=r['task_id'];n=1
 while (P/'runs'/tid/f'a{n}'/'guard.json').exists():n+=1
 attempt=f'a{n}';out=f'{E}/runs/{tid}/{attempt}';dest=P/'runs'/tid/attempt;dest.mkdir(parents=True,exist_ok=True)
 code="import shutil; assert shutil.disk_usage('/opt').free>=50.5*2**30, 'disk below admission'"
 prep=subprocess.run(SSH+[shlex.join(['python3','-c',code])],capture_output=True,text=True)
 if prep.returncode:print(prep.stderr,flush=True);break
 cmd=['python3',R+'/tools/baseline_retest32_server1_guard_receipt.py','--owner','v3-sample20/server1/'+tid,'--wait','600','--timeout','3600','--receipt',out+'/guard.json','--','bash',E+'/launch.sh',r['key'][0],E+'/bootstrap.py','--task',E+'/tasks/'+tid+'/task.json','--out',out]
 start=time.time();print('START',i,r['key'],attempt,flush=True)
 with (dest/'console.log').open('w') as f:
  rc=subprocess.run(SSH+['mkdir -p '+shlex.quote(out)+' && cd '+shlex.quote(R)+' && '+shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT).returncode
 # Collect all output, including raw trace; local tar is a retained complete receipt.
 archive=dest/'receipt.tar'
 with archive.open('wb') as f:
  subprocess.run(SSH+[shlex.join(['tar','-C',out,'-cf','-','.'])],stdout=f,check=True)
 subprocess.run(['tar','-xf',str(archive),'-C',str(dest)],check=True)
 status=json.loads((dest/'measurement.json').read_text()) if (dest/'measurement.json').exists() else {}
 summary={'index':i,'key':r['key'],'attempt':attempt,'returncode':rc,'wall_seconds':time.time()-start,'status':status.get('status'),'error':status.get('error'),'at':time.time()}
 with (P/'journal.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
 print('DONE',json.dumps({**summary,'error':str(summary['error'])[:400] if summary['error'] else None}),flush=True)
