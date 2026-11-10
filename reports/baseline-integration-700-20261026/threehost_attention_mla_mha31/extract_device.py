from pathlib import Path
import subprocess,re,json,hashlib
P=Path(__file__).resolve().parent
for p in P.glob('**/launch_*/native.json'):
 d=p.parent;bpath=next(d.glob('*.so'));b=bpath.read_bytes()
 for arg,name in [('-SW','ELF_SECTIONS.txt'),('-sW','ELF_SYMBOLS.txt')]:
  r=subprocess.run(['readelf',arg,str(bpath)],capture_output=True,text=True);(d/name).write_text(r.stdout+r.stderr)
 rows=[]
 for line in (d/'ELF_SECTIONS.txt').read_text().splitlines():
  m=re.search(r'\[\s*\d+\]\s+(\S+)\s+PROGBITS\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)',line)
  if m and m[1] in ['.aicore_binary','__aicore_rel_binary']:
   data=b[int(m[2],16):int(m[2],16)+int(m[3],16)];f=d/(m[1]+'.bin');f.write_bytes(data);rows.append({'section':m[1],'offset':int(m[2],16),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'file':f.name})
 (d/'DEVICE_BINARY.json').write_text(json.dumps(rows,indent=2))
 print(str(d),len(rows))
