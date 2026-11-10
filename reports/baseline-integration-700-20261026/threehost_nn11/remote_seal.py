from pathlib import Path
import hashlib,json,datetime
P=Path(__file__).resolve().parent
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows={}
for q in P.iterdir():
 if not q.is_dir() or not (q/'GUARD_RESULT.json').exists():continue
 rows[q.name]={'guard':json.loads((q/'GUARD_RESULT.json').read_text()),'artifacts':{str(f.relative_to(q)):{'sha256':h(f),'bytes':f.stat().st_size} for f in q.rglob('*') if f.is_file() and f.suffix in ['.pt','.json','.jsonl','.txt'] and '/.cache/' not in str(f)}}
files=[p for p in P.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.json','.so','.o'] and p.name!='REMOTE_SEAL.json']
(P/'REMOTE_SEAL.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':rows,'entry_and_deployment_sha256':{str(f):h(f) for f in files},'accepted_added':0},indent=2));print('sealed',len(rows),'attempts')
