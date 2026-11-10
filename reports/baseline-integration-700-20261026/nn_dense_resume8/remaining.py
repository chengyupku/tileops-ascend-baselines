from pathlib import Path
import json,subprocess,sys,datetime
P=Path(__file__).resolve().parent
first=json.loads((P/'case01/FROZEN.json').read_text());assert first['numerical_pass']
keys=[r['case_id'] for r in json.loads((P/'SCOPE.json').read_text()) if r['case_id']!=first['case_id']]
def save(phase):
 d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'numerical_pass':len(list(P.glob('case*/FROZEN.json'))),'registered':0,'parent_accepted_added':0,'frozen_evidence':[str(p) for p in P.glob('case*/FROZEN.json')]}
 for n in ['HEARTBEAT.json','INTERIM_DELIVERY.json']:(P/n).write_text(json.dumps(d,indent=2))
save('first_key_gate_passed')
for i,key in enumerate(keys,2):
 attempt=f'case{i:02d}'
 subprocess.run([sys.executable,str(P/'launch.py'),key,attempt],check=True)
 g=json.loads((P/attempt/'GUARD_RESULT.json').read_text())
 if g['returncode']!=0:save('stopped_after_guard_or_checker_failure');break
 r=subprocess.run([sys.executable,str(P/'freeze.py'),attempt])
 if r.returncode:save('stopped_after_evidence_audit_failure');break
 save('frozen_'+attempt)
