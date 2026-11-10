from pathlib import Path
import shutil,json,hashlib
P=Path(__file__).resolve().parent
for r in (P/'runs').iterdir():
 artifacts=[]
 for f in (r/'TRITON_CACHE_DIR').rglob('*'):
  if not f.is_file() or f.suffix not in ('.json','.npubin','.ttir'):continue
  b=f.read_bytes();out=r/'native_artifacts'/f.parent.name/f.name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
  artifacts.append({'original_path':str(f),'saved_path':str(out.relative_to(P)),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
 (r/'NATIVE_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2))
