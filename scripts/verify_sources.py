"""Verify exported source integrity without importing/executing benchmark code."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'catalog/source-manifest.json').read_text());bad=[]
for r in rows:
 p=root/r['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:bad.append(r['path'])
print(json.dumps({'checked':len(rows),'mismatches':bad},indent=2))
raise SystemExit(bool(bad))
