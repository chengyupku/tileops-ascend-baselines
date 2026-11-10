"""CPU orchestration; every device process is a foreground npu-guard child."""
import json, subprocess, time, datetime, hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
W=Path('/root/dyq/workspace')
order=['throughput-4k-complex64','shortconv-64-complex64','fnet-512-complex64','spectrogram-1k-complex64','spectrogram-1k-b300-complex64','longconv-16k-complex64','longconv-64k-complex64']
def write(p,x): p.write_text(json.dumps(x,indent=2))
def read(p): return json.loads(p.read_text()) if p.exists() else {}
def status(p):
 r=subprocess.run(['npu-guard','status','--json'],capture_output=True,text=True);p.write_text(r.stdout);return r.returncode
rows=[]
for case in order:
 d=P/case;d.mkdir(exist_ok=True)
 assert not (d/'COMMAND.json').exists(), 'No repeat attempts'
 write(P/'HEARTBEAT.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'guard_queue_or_run','case_id':case,'completed':len(rows)})
 status(d/'STATUS_BEFORE.json')
 cmd=['npu-guard','run','--owner','baseline700-20261026/fft_remaining_c64_7','--wait','1200','--timeout','180','--','.tools/baselines/baseline-run','ops-math','python',str(P/'runner.py'),case]
 write(d/'COMMAND.json',cmd)
 start=time.monotonic()
 with (d/'guard.log').open('w') as log:
  result=subprocess.run(cmd,cwd=W,stdout=log,stderr=subprocess.STDOUT)
 write(d/'GUARD_RESULT.json',{'returncode':result.returncode,'elapsed_s':time.monotonic()-start})
 status(d/'STATUS_AFTER.json')
 c=read(d/'checker.json');child=read(d/'CHILD_RESULT.json')
 ok=result.returncode==0 and child.get('returncode')==0 and c.get('status')=='verified'
 # Canonical checker status vocabulary is preserved in the row.
 if c.get('status')=='passed' and result.returncode==0 and child.get('returncode')==0: ok=True
 row={'baseline':'ops-math','operator':'FFTC2CFwdOp','case_id':case,'status':'verified_unregistered' if ok else 'failed','reason':c.get('reason',c.get('error','Canonical full checker passed' if ok else 'See raw checker/guard receipt')),'checker':str(d/'checker.json'),'guard':str(d/'GUARD_RESULT.json'),'checker_status':c.get('status'),'child_returncode':child.get('returncode'),'guard_returncode':result.returncode,'registered':False,'new_parent_acceptances':0}
 rows.append(row);write(P/'RESULT.json',{'rows':rows,'in_progress':True,'new_parent_acceptances':0})
 print(json.dumps(row),flush=True)
 if result.returncode in [75,76,78]:
  for pending in order[len(rows):]: rows.append({'baseline':'ops-math','operator':'FFTC2CFwdOp','case_id':pending,'status':'blocked','reason':'Not run: stopped after guard refusal/interference '+str(result.returncode),'checker':None,'guard':None,'registered':False})
  break
write(P/'RESULT.json',{'rows':rows,'in_progress':False,'new_parent_acceptances':0})
