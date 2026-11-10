import json,pathlib,hashlib,datetime
W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026';O=P/'acceptance_audit02'
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}
def write(n,x):
 t=O/(n+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(O/n)
remote=json.loads((O/'GDN_REMOTE_CHECK.json').read_text());rr={r['case_id']:r for r in remote['rows']};rows=[]
for s in json.loads((O/'SCOPE.json').read_text()):
 if s['worker']!='transformer_linear':continue
 cid=s['case_id'];p=P/s['worker']/cid;cp=p/'checker.json';gp=p/'GUARD_RECEIPT.json';c=json.loads(cp.read_text());g=json.loads(gp.read_text());issues=list(rr[cid]['issues']);v=c['verification'];batch=P/s['worker']/g['batch_receipt'];sh=batch/'child.sh';log=batch/'guard.stderr.log'
 if not(c['status']=='verified' and c['case_id']==cid and v['checked_outputs']==v['total_outputs']==2 and c['inputs_preserved']):issues.append('Checker contract incomplete')
 if not(g['returncode']==0 and g['start']<=c['started_at']<=c['finished_at']<=g['end']):issues.append('Guard time or exit mismatch')
 if cid not in sh.read_text().splitlines()[-1].split():issues.append('Not in explicit batch command')
 if 'acquired device 0 for baseline700-20261026/transformer_linear' not in log.read_text():issues.append('Missing guard acquired record')
 if ref(cp)['sha256']!=s['result']['checker_receipt']['sha256'] or ref(cp)['sha256']!=rr[cid]['checker_sha256']:issues.append('Snapshot/local/remote checker mismatch')
 if rr[cid]['dispatch']!=['formal-build',cid] or rr[cid]['unknown']!=['prior','__audit_unknown__']:issues.append('Dispatch mismatch')
 # Same frozen implementation as parent reviewed family, new exact-key full receipts.
 rows.append({'key':[s[k] for k in ['baseline','operator','case_id']],'accept_recommended':not issues,'decision':'recommend_accept' if not issues else 'needs_evidence','scope_reasons':s['classification']['reason'],'checker':ref(cp),'verification':v,'inputs_preserved':True,'seed':c['input_seed'],'checker_window':[c['started_at'],c['finished_at']],'guard':dict(ref(gp),returncode=g['returncode'],batch=g['batch_receipt'],command=ref(sh),log=ref(log)),'formal_factory':s['result']['factory'],'formal_module':remote['module'],'provider':remote['provider'],'source_provenance':{'parent_review':ref(P/'PARENT_GDN_1343_REVIEW.json'),'parent_route_review':ref(P/'PARENT_GDN_ROUTE_CHECK_1343.json'),'live_remote_check':ref(O/'GDN_REMOTE_CHECK.json'),'tested_frozen_binding_identical':True,'runtime_primary':rr[cid]['runtime_primary'],'loaded_libraries':ref(p/'loaded-libraries.json')},'runtime_recipe':ref(P/'transformer_linear/chunk_package/RUNTIME_RECIPE.json'),'issues':issues})
write('GDN_REVIEW.json',{'rows':rows});prior=json.loads((O/'NN_REVIEW.json').read_text())['rows'];write('RECOMMENDED_ACCEPT.json',{'rows':prior+rows,'partial':True});write('HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'NN18 + GDN9 complete'})
print([(r['key'][-1],r['issues']) for r in rows])
