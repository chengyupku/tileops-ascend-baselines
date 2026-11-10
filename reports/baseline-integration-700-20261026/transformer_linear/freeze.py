from pathlib import Path
import json,shutil,hashlib
W=Path(__file__).resolve().parent;F=W/'frozen_v1';F.mkdir(exist_ok=True)
for n in ['CONTRACTS.json','CHUNK_CASES.json']:
 if not (F/n).exists():shutil.copyfile(W/n,F/n)
P=F/'chunk_package';P.mkdir(exist_ok=True)
for n in ['binding.py','direct_acl.py','entry.py','RUNTIME_RECIPE.json','RUNTIME_PINS.json']:
 if not (P/n).exists():shutil.copyfile(W/'chunk_package'/n,P/n)
proof=[]
for i in json.loads((W/'REGISTERED.json').read_text()):
 c=json.loads((W/i/'checker.json').read_text())
 for n in ['binding.py','direct_acl.py','entry.py']:
  before=c['source_sha256'][str(W/'chunk_package'/n)];after=hashlib.sha256((P/n).read_bytes()).hexdigest();assert before==after
  proof.append({'case_id':i,'file':n,'verified_sha256':before,'frozen_sha256':after,'equal':True})
(W/'FREEZE_PROOF.json').write_text(json.dumps({'version':'frozen_v1','proof':proof,'runtime_sha256':hashlib.sha256((P/'RUNTIME_RECIPE.json').read_bytes()).hexdigest()},indent=2))
s=(W/'register.py').read_text().replace("package=PRIVATE/'chunk_package'","package=PRIVATE/'frozen_v1/chunk_package'");t=W/'register.tmp';t.write_text(s);t.replace(W/'register.py')
print('frozen',len(proof))
