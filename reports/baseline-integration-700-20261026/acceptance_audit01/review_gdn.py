import pathlib,json,hashlib,datetime
O=pathlib.Path(__file__).parent;B=O.parent;E=json.load(open(O/'EVIDENCE_INDEX.json'));S=json.load(open(O/'SCOPE.json'));H=json.load(open(O/'server2_hashes.json'));T=json.load(open(O/'server2_static.json'));rows=[]
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for r,x in zip(S,E):
 if r['worker']!='transformer_linear':continue
 c=x['checker_summary'];g=json.load(open(x['guard']['path']));batch=B/'transformer_linear'/g['batch_receipt'];assert r['case_id'] in (batch/'child.sh').read_text();assert g['returncode']==0 and g['start']<c['started_at']<c['finished_at']<g['end'];v=c['verification'];assert v['checked_outputs']==v['total_outputs']==2 and c['inputs_preserved'] and not c['oracle_source_changed']
 for which in ['checker','guard']:
  remote=x[which]['path'].replace('/root/dyq/workspace','/opt/tileops-all-pairs/workspace');assert H[remote]['sha256']==x[which]['sha256']
 for p,h in x['source'].items():
  if p in H:assert H[p]['sha256']==h,(p,h,H[p])
 old=json.load(open(B/'PARENT_GDN_ROUTE_CHECK_1343.json'))
 for p,h in old['hashes'].items():
  if p in H:assert H[p]['sha256']==h
 keys=json.loads(T['/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_linear/REGISTERED.json']['text']);assert r['case_id'] in keys
 rows.append({'key':x['key'],'accept_recommended':True,'scope_reasons':r['classification'],'checker':x['checker'],'guard':x['guard'],'guard_returncode':g['returncode'],'batch_evidence':{'path':str(batch/'child.sh'),'sha256':digest(batch/'child.sh')},'formal_factory':r['result']['factory'],'module_sha256':old['hashes']['/opt/tileops-all-pairs/workspace/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/integration700_transformer_linear_gdn.py'],'source_provenance':str(B/'PARENT_GDN_1343_REVIEW.json'),'review':'Complete y/state checker and inputs preserved; exact key present in live REGISTERED. Same original full chunk API binding/direct transport and primary runtime hashes as previous review, preserving previous provider fallback; exact clean bounded batch contains checker interval.','issues':[]})
def put(n,j):
 p=O/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
put('GDN_RECOMMENDED_ACCEPT.json',{'rows':rows});allrows=json.load(open(O/'RECOMMENDED_ACCEPT.json'))['rows'];put('RECOMMENDED_ACCEPT.json',{'rows':[r for r in allrows if r['key'][0]!='ops-transformer']+rows});print('GDN',len(rows))
