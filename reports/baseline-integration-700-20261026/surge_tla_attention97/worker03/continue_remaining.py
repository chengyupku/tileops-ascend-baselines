from pathlib import Path
import os,time,datetime,json,subprocess,sys
from orchestrate import run,beat,P
keys=json.loads((P/'CONTINUATION_PLAN.json').read_text())['keys']
limit=datetime.datetime(2026,10,27,5,31,tzinfo=datetime.timezone.utc).timestamp()
os.environ['ATTEMPT_LABEL']='-continuation01'
os.environ['CHANGED_CONDITION']='Exact key never executed; authorized FIFO guard wait increased from 120 to 600 seconds'
for i,key in enumerate(keys[1:],1):
 if time.time()+1220>limit:
  (P/'CONTINUATION_DEADLINE_REMAINING.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unsubmitted':keys[i:],'reason':'Cannot fit authorized wait600/run600 plus collection before deadline'},indent=2));break
 os.environ['HOST_PREFERENCE']=['openlibing-DevEnv-661495','openlibing-DevEnv-055409'][i%2]
 run(key[2])
 subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
(P/'CONTINUATION_BATCH_COMPLETE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
