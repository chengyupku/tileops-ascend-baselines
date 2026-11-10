from pathlib import Path
import hashlib,json
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/gdn_oracle_probe6'); O=P/'runs/probe01--GDNFwdOp--prefill-narrow-state-bfloat16/output'
paths=set()
for line in (O/'maps.txt').read_text().splitlines():
 fields=line.split();p=Path(fields[-1])
 if p.is_absolute() and p.is_file() and ('.so' in p.name or 'python' in p.name):paths.add(p)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[dict(path=str(p),size=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)]
(P/'BINARY_IDENTITIES.json').write_text(json.dumps(dict(source='post-exit hashes of file paths mapped by actual process; no device imports',binaries=rows),indent=2))
print(json.dumps(dict(binary_count=len(rows))))
