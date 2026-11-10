from pathlib import Path
import json,hashlib,time
P=Path(__file__).resolve().parent
r=json.loads((P/'HOST_FREEZE.json').read_text());checked=[];diff=[]
for path,old in r['files'].items():
 f=Path(path)
 if not f.exists():diff.append(dict(path=path,reason='missing'));continue
 with f.open('rb') as h:actual=hashlib.file_digest(h,'sha256').hexdigest()
 checked.append(path)
 if actual!=old['sha256']:diff.append(dict(path=path,before=old['sha256'],after=actual))
(P/'POST_AUDIT.json').write_text(json.dumps(dict(at=time.time(),checked_files=len(checked),changes=diff),indent=2));print(json.dumps(dict(checked=len(checked),changes=diff)))
