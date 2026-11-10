from pathlib import Path
import json,datetime,sys
P=Path(__file__).resolve().parent
rows=[]
for d in P.iterdir():
 if not d.is_dir() or not (d/'GUARD_RESULT.json').is_file():continue
 g=json.loads((d/'GUARD_RESULT.json').read_text());c=json.loads((d/'checker.json').read_text()) if (d/'checker.json').exists() else {}
 row={'run':d.name,'exactkey':[c.get('library'),c.get('op'),c.get('case_id')],'guard_returncode':g['returncode'],'checker_status':c.get('status'),'verification':c.get('verification'),'inputs_preserved':c.get('inputs_preserved'),'error':c.get('error'),'checker':str((d/'checker.json').relative_to(P)) if c else None,'audit':str((d/'VALIDATION_AUDIT.json').relative_to(P)) if (d/'VALIDATION_AUDIT.json').exists() else None,'accepted_added':0}
 rows.append(row)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
result={'utc':now,'accepted_added':0,'registered_added':0,'runs':rows,'verified_exactkeys':[r['exactkey'] for r in rows if r['checker_status']=='verified' and r['guard_returncode']==0],'phase':sys.argv[1] if len(sys.argv)>1 else 'running'}
(P/'RESULT.json').write_text(json.dumps(result,indent=2));(P/'HEARTBEAT.json').write_text(json.dumps({'utc':now,'phase':result['phase'],'accepted_added':0},indent=2));print(json.dumps(result))
