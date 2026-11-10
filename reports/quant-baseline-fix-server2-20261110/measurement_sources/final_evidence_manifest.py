"""Hash retained accepted raw artifacts locally; this manifest is not exported."""
import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'RESULTS.json').read_text());files={}
for r in rows:
 if r['status']!='formal_raw_verified':continue
 root=P/f"remote/runs/{r['index']}/{r['variant']}/{r['attempt']}"
 for f in sorted(root.rglob('*')):
  if f.is_file():files[str(f.relative_to(P))]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
text=json.dumps({'accepted_variants':sum(x['status']=='formal_raw_verified' for x in rows),'files':files},indent=2)+'\n'
out=P/'RAW_FILE_MANIFEST.private.json';out.write_text(text)
(P/'RAW_FILE_MANIFEST.private.sha256').write_text(hashlib.sha256(text.encode()).hexdigest()+'  '+out.name+'\n')
print('Accepted raw files hashed',len(files),sum(x['bytes'] for x in files.values()))
