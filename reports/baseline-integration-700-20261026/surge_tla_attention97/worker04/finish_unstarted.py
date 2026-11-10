from pathlib import Path
import json,subprocess,sys,time,datetime
from orchestrate import statuses,beat
P=Path(__file__).resolve().parent
end=datetime.datetime(2026,10,27,5,23,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<end:
 subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
 rows=json.loads((P/'RESULT.json').read_text());pending=[r for r in rows if r['key'][1]=='GQADenseFwdOp' and r['status'] in ['guard_not_started','guard_visibility_failure']]
 if not pending:break
 active=[json.loads(f.read_text()) for f in (P/'attempts').glob('*/command.json') if not (f.parent/'guard.json').exists()]
 pending=[r for r in pending if tuple(r['key']) not in {tuple(a['key']) for a in active}]
 if not pending:break
 tag='idle-opportunity-'+str(int(time.time()));st=statuses(tag);choice=None
 for row in pending:
  started_hosts={json.loads(f.read_text())['host'] for f in (P/'attempts').glob('*/guard.json') if json.loads(f.read_text())['key']==row['key'] and list(f.parent.rglob('GUARD_RUNNING.json'))}
  host=next((h for h in ['openlibing-DevEnv-055409','openlibing-DevEnv-661495','local'] if st.get(h)==0 and h not in {a['host'] for a in active} and h not in started_hosts),None)
  if host:choice=(row,host);break
 host=choice[1] if choice else None
 if host:
  cid=choice[0]['key'][2];subprocess.run([sys.executable,str(P/'orchestrate.py'),cid,host,tag,'No prior workload start on this host for this exact key; newly observed compatible host idle. Evidence: '+tag],check=True)
 else:
  beat(pending[0]['key'],'waiting_for_new_idle_opportunity',tag,'All three hosts busy/unknown; no repeat submission')
  time.sleep(30)
subprocess.run([sys.executable,str(P/'summarize.py')],check=True)
