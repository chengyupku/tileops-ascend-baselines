from pathlib import Path
import json,hashlib,datetime
P=Path(__file__).resolve().parent;W=P.parents[2];pins=json.loads((P/'package_local/RUNTIME_PINS.json').read_text())['files'];rows=[]
for x in pins:
 p=Path(x['path']);actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None;rows.append({'path':str(p),'sha256':actual,'prior_sha256':x['sha256'],'matches_prior':actual==x['sha256']})
for root in [W/'.venv/lib/python3.12/site-packages/fla',W/'.venv/lib/python3.12/site-packages/triton']:
 for p in root.rglob('*.py'):rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(P/'LOCAL_DEPENDENCY_FREEZE.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'original_native_pin_mismatches':sum(x.get('matches_prior') is False for x in rows)},indent=2));print('files',len(rows),'native_mismatches',sum(x.get('matches_prior') is False for x in rows))
