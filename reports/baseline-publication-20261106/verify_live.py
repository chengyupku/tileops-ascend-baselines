import json,urllib.request,datetime,hashlib,gzip
from pathlib import Path
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));p=Path('reports/baseline-publication-20261106');release=json.loads((p/'RELEASE.json').read_text());url='https://yyttt6.github.io/TileOPs.github.io/benchmarks/baseline-cache.json?release='+release['website_commit']
req=urllib.request.Request(url,headers={'Accept-Encoding':'gzip'})
with opener.open(req,timeout=30) as r:
 raw=r.read()
 if r.headers.get('Content-Encoding')=='gzip':raw=gzip.decompress(raw)
d=json.loads(raw);s=json.loads(Path('/tmp/tileops-site-torch2208-20261106/data/baseline-cache/snapshot.json').read_text());key=lambda x:(x['library'],x['op'],x['case']);a={key(x):x for x in s['rows']};b={key(x):x for x in d['rows']};assert len(a)==len(b)==7758,(len(a),len(b))
for k,x in a.items():
 for f in ['latency_us','measurement_id','samples','protocol','workload']:assert x[f]==b[k][f],(k,f)
v={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'baseline_rows_identical_to_CI':7758,'new_torch_rows':2208,'live_snapshot_sha256':hashlib.sha256(raw).hexdigest(),'tileops_cases_preserved':len({(x['op'],x['case']) for x in d['rows'] if x['tileops_latency_us'] is not None})};p.joinpath('LIVE_VERIFICATION.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
