from pathlib import Path
import json,hashlib,subprocess
P=Path(__file__).resolve().parent; R=P/'runs/serving-kv512-float16-original-default-attempt02'
rows={}
for line in (R/'maps.txt').read_text().splitlines():
 a=line.split(None,5)
 if len(a)!=6 or not a[-1].startswith('/'):continue
 f=Path(a[-1])
 if not any(x in f.name for x in ['tilelang','tvm','runtime','ascendcl','torch_npu','cython_wrapper']):continue
 if not f.is_file():continue
 if str(f) not in rows:
  s=f.stat();b=f.read_bytes();rows[str(f)]={'sha256':hashlib.sha256(b).hexdigest(),'inode':s.st_ino,'device':s.st_dev,'elf_machine':int.from_bytes(b[18:20],'little') if b[:4]==b'\x7fELF' else None,'maps':[]}
 rows[str(f)]['maps'].append(line)
(R/'loaded_dependencies.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
