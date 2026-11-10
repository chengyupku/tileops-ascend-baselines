import json,urllib.request,gzip,hashlib,datetime
from pathlib import Path
p=Path('reports/baseline24-publication-20261108');o=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:15014'}))
req=urllib.request.Request('https://yyttt6.github.io/TileOPs.github.io/benchmarks/baseline-cache.json?baseline24=8cc05d7b',headers={'Accept-Encoding':'gzip'})
with o.open(req,timeout=45) as r:
 b=r.read()
 if r.headers.get('Content-Encoding')=='gzip':b=gzip.decompress(b)
d=json.loads(b);v={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(b).hexdigest(),'nightly':d['nightly'],'baseline_rows':len(d['rows'])};p.joinpath('LIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
expected=json.loads((p/'LOCAL_VERIFICATION.json').read_text());assert v['sha256']==expected['rendered_sha256'], 'Live data does not yet match audited render'
