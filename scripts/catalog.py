"""Read-only catalog, no torch/NPU import."""
import argparse, collections, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command',choices=['summary','list']);p.add_argument('--library');p.add_argument('--status');a=p.parse_args()
rows=json.loads((Path(__file__).resolve().parents[1]/'catalog/results.json').read_text())
rows=[r for r in rows if (not a.library or r['key'][0]==a.library) and (not a.status or r['status']==a.status)]
if a.command=='summary':
 print(json.dumps({c:dict(collections.Counter(r['status'] for r in rows if r['campaign']==c)) for c in sorted({r['campaign'] for r in rows})},indent=2))
else:
 for r in rows: print('\t'.join([r['campaign'],*r['key'],r['status']]))
