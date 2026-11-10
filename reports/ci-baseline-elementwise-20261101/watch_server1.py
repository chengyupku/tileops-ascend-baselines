"""Read-only half-hour checks, starting at the user's requested first window."""
import datetime,json,subprocess,shlex,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
remote=r'''
import json,time,hashlib
from pathlib import Path
root=Path('/opt/tileops-baseline-cache/workspace/reports/baseline-retest32-20261030')
state=json.loads((root/'server1/STATE.json').read_text())
assigned=json.loads((root/'server1_ASSIGNMENTS.json').read_text())['pairs']
expected={tuple(x['key']) for x in assigned}
rows=[]
with (root/'server1/CACHE.jsonl').open() as f:
 for line in f:
  try:r=json.loads(line)
  except json.JSONDecodeError:continue
  rows.append(r)
cached={tuple(r['key']) for r in rows if r.get('status')=='cached' and r.get('performance_valid') is True}
missing=expected-cached
print(json.dumps({'at':time.time(),'phase':state.get('phase'),'assigned':len(expected),'cached':len(expected&cached),'missing':[list(x) for x in sorted(missing)],'extra':len(cached-expected),'rows':len(rows),'identity':state.get('identity'),'pending_review':state.get('pending_review'),'complete':not missing and not(cached-expected)},ensure_ascii=False))
'''
next_at=datetime.datetime.fromisoformat('2026-11-01T03:34:52+00:00').timestamp()
print('First server1 query not before '+datetime.datetime.fromtimestamp(next_at,datetime.timezone.utc).isoformat(),flush=True)
while True:
 while time.time()<next_at:time.sleep(min(30,next_at-time.time()))
 record={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','openlibing-DevEnv-661495','python3 -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=180)
  record.update(exit_code=r.returncode)
  if r.returncode==0:record['result']=json.loads(r.stdout)
  else:record['error']=r.stderr[-2000:]
 except Exception as e:record['error']=repr(e)
 with (ROOT/'server1-checks.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 (ROOT/'server1-last-check.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(record,ensure_ascii=False),flush=True)
 if record.get('result',{}).get('complete'):break
 next_at=max(next_at+1800,time.time()+1)
