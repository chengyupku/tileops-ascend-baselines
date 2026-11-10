import pathlib,json,hashlib,sys
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/sgl_linalg36'
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((P/'DELIVERY.json').read_text());rows=[];hashes={};errors=[]
for path,expected in json.loads((P/'SOURCE_HASHES.json').read_text()).items():
 p=pathlib.Path(path);p=p if p.is_absolute() else W/p
 actual=h(p);hashes[str(p)]={'expected':expected,'actual':actual};assert actual==expected,(p,actual)
for r in d['mapping']:
 cp=P/r['checker_receipt'];gp=P/r['guard_receipt'];c=json.loads(cp.read_text());g=json.loads(gp.read_text());route=cp.parent/'ROUTE.json';rt=json.loads(route.read_text())
 assert c['status']=='verified' and g['exit_code']==0
 assert [c['op'],c['case_id'],c['library']]==[r['operator'],r['case_id'],r['baseline']]
 assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert c['inputs_preserved'] and not c['oracle_source_changed'] and not c['provider_source_changed']
 assert rt['adapter_hash']==d['adapter_sha256']
 for f,v in c['source_sha256'].items():
  q=pathlib.Path(f);q=q if q.is_absolute() else W/q
  assert h(q)==v,(f,'source changed')
 for f,v in c['oracle_source_sha256'].items():assert h(W/f)==v,(f,'oracle changed')
 assert h(W/'TileOPs/scripts/ascend_pair_check.py')==rt['checker_hash']
 rows.append(dict(key=[r['baseline'],r['operator'],r['case_id']],checker={'path':str(cp),'sha256':h(cp)},guard={'path':str(gp),'sha256':h(gp),'exit_code':g['exit_code'],'command':g['command']},route={'path':str(route),'sha256':h(route),**rt},seed=c['input_seed'],case_snapshot=c['case_snapshot'],observed_calls=c['observed_calls'],verification=c['verification'],oracle_source_sha256=c['oracle_source_sha256'],inputs_preserved=c['inputs_preserved']))
print(json.dumps(dict(rows=rows,hashes=hashes,adapter_sha256=d['adapter_sha256'],source_directory=d['source_directory'],shared_library=d['shared_library'],interpreter=d['interpreter'])))
