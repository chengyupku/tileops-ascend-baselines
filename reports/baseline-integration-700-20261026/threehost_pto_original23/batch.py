from pathlib import Path
import subprocess,json,sys,time
P=Path(__file__).resolve().parent
scope=json.loads((P/'SCOPE.json').read_text());history=json.loads((P.parent/'threehost_pto_historical23/REQUALIFICATION.json').read_text())['rows'];sizes={tuple(r['key']):sum(__import__('math').prod(v[0]) for v in r['historical_checker']['case_snapshot']['parameters'].values() if isinstance(v,list) and len(v)==2 and isinstance(v[0],list)) for r in history}
scope.sort(key=lambda r:sizes[(r['baseline'],r['operator'],r['case_id'])])
for r in scope:
 key=[r['baseline'],r['operator'],r['case_id']]
 status=json.loads((P/'RESULT.json').read_text());done={tuple(x['key']) for x in status['rows'] if x['status']=='full_checker_native_verified'}
 if tuple(key) in done:continue
 host='server1';tag='_'.join(key)
 subprocess.run([sys.executable,str(P/'launch.py'),host,*key],check=True)
 guard=json.loads((P/(tag+'.guard.json')).read_text())
 if guard['returncode'] in (75,76,78):
  for h in ['server2']:
   out=subprocess.run([sys.executable,str(P/'transport.py'),h,'npu-guard status --json'],capture_output=True,text=True)
   (P/(tag+'.'+h+'.status.json')).write_text(out.stdout)
  out=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);(P/(tag+'.local.status.json')).write_text(out.stdout)
  host='server2';subprocess.run([sys.executable,str(P/'launch.py'),host,*key,'_migrated'],check=True)
 subprocess.run([sys.executable,str(P/'pull.py'),host],check=True)
 subprocess.run([sys.executable,str(P/'checkpoint.py')],check=True)
print('BATCH COMPLETE')
