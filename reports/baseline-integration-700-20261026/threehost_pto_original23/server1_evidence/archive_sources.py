from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parent;r=[]
for run in (P/'runs').iterdir():
 f=run/'subprocess.jsonl'
 if not f.exists():continue
 for line in f.read_text().splitlines():
  call=json.loads(line);argv=call['argv']
  if not isinstance(argv,list) or not any('bisheng' in str(x) for x in argv[:1]):continue
  for x in argv:
   p=Path(x)
   if p.suffix=='.cpp' and p.is_file():
    h=hashlib.sha256(p.read_bytes()).hexdigest();d=P/'compile_sources'/h;d.parent.mkdir(exist_ok=True);shutil.copyfile(p,d);r.append(dict(run=str(run),original=str(p),sha256=h,saved=str(d),argv=argv))
(P/'COMPILE_SOURCES.json').write_text(json.dumps(r,indent=2));print(len(r))
