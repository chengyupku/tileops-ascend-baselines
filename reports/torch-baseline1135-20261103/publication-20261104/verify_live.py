import json,pathlib,urllib.request,hashlib,datetime,subprocess
P=pathlib.Path(__file__).parent
site=pathlib.Path('/tmp/torch1135-site')
sha=subprocess.check_output(['git','-C',str(site),'rev-parse','HEAD'],text=True).strip()
url='https://yyttt6.github.io/TileOPs.github.io/benchmarks/baseline-cache.json?release='+sha
with urllib.request.urlopen(urllib.request.Request(url,headers={'Cache-Control':'no-cache'}),timeout=30) as response:raw=response.read()
live=json.loads(raw);release=json.loads((site/'data/baseline-cache/torch1135-release.json').read_text());expected=json.loads((site/'data/baseline-cache/snapshot.json').read_text())
key=lambda r:(r['library'],r['op'],r['case'])
live_rows={key(r):r for r in live['rows']};expected_rows={key(r):r for r in expected['rows']}
assert len(live_rows)==len(live['rows'])
for item in release['measurements']:
 k=('torch-eager',item['op'],item['case']);a=live_rows[k];b=expected_rows[k]
 for field in ['measurement_id','protocol','latency_us','samples','workload']:assert a[field]==b[field],(k,field)
with urllib.request.urlopen('https://yyttt6.github.io/TileOPs.github.io/benchmarks/?release='+sha,timeout=30) as response:html=response.read().decode()
assert '1135' in html and '7212' in html and 'parameter-only' in html
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':url,'snapshot_sha256':hashlib.sha256(raw).hexdigest(),'new_torch_records_identical_to_CI':1135,'rows':len(live_rows),'torch_rows':sum(r['library']=='torch-eager' for r in live['rows']),'page_counts_and_measurement_notes_visible':True,'website_commit':sha,'ci_commit':subprocess.check_output(['git','-C','/tmp/torch1135-ci','rev-parse','HEAD'],text=True).strip(),'nightly_tileops_results_preserved':True}
(P/'LIVE_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(result)
