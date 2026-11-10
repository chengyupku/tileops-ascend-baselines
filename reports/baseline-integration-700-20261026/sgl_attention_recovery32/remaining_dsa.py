from pathlib import Path
import subprocess,sys,json,datetime
P=Path(__file__).resolve().parent
for i in [20,16,9,10,11,13,14,15,18,19,21,17]:
 label=f'dsa07_{i:02d}'
 subprocess.run([sys.executable,str(P/'launch.py'),label,str(i)],check=True)
 subprocess.run([sys.executable,str(P/'refresh.py')],check=True)
 g=json.loads((P/'attempts'/label/'GUARD_RESULT.json').read_text());c=P/'runs'/label/'checker.json';d=json.loads(c.read_text()) if c.exists() else {}
 (P/'HEARTBEAT').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+label+' completed\n')
 if g['exit_code'] or d.get('status')!='verified':
  (P/'DSA_STOP.json').write_text(json.dumps({'index':i,'attempt':label,'guard_exit':g['exit_code'],'checker_status':d.get('status'),'error':d.get('error'),'reason':'Stop common DSA path at first failure; no unchanged retry'},indent=2));break
