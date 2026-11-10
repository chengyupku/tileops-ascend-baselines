from pathlib import Path
P=Path(__file__).resolve().parent
r=(P/'run_isnan.py').read_text()
r=r.replace("receipt=out/(str(seq)+'.dispatch.json')", """actual={}
 for line in r['maps'].splitlines():
  parts=line.split(None,5)
  if len(parts)==6 and parts[5].startswith('/'):
   path=Path(parts[5])
   if path.is_file() and any(s in str(path) for s in ('torch','tilelang','tvm','Ascend','sdk-920b2','libpython')):
    actual[str(path)]=dict(sha256=digest(path),inode=path.stat().st_ino,size=path.stat().st_size)
 (out/'actual_runtime_pins.json').write_text(json.dumps(actual,indent=2))
 receipt=out/(str(seq)+'.dispatch.json')""")
compile(r,'runner','exec');(P/'run_isnan.py').write_text(r)
