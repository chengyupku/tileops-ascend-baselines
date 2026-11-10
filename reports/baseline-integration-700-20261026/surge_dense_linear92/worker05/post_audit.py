from pathlib import Path
import hashlib,json,datetime
W=Path(__file__).resolve().parent
for a in sorted(W.glob('attempt*')):
 if not a.is_dir() or not (a/'FROZEN.json').exists():continue
 frozen=json.loads((a/'FROZEN.json').read_text())['files'];changed=[];missing=[]
 for name,pin in frozen.items():
  p=Path(name)
  if not p.is_file():missing.append(name)
  elif hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']:changed.append(name)
 captures={}
 for name in ['inputs.pt','outputs.pt','bound_inputs.pt','optional_aqk_akk.pt']:
  p=a/name;captures[name]=dict(exists=p.exists(),bytes=p.stat().st_size if p.exists() else None,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity_phase='post-use comparison against actual pre-use pins',changed=changed,missing=missing,captures=captures)
 (a/'POST_AUDIT.json').write_text(json.dumps(result,indent=2));print(a.name,json.dumps(result))
