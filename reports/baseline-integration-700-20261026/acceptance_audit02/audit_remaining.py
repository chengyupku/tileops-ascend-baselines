import json,pathlib,hashlib,datetime,re
W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026';O=P/'acceptance_audit02'
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}
def write(n,x):
 t=O/(n+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(O/n)
def dt(s):return datetime.datetime.fromisoformat(s)
dense=json.loads((O/'DENSE_REMOTE_CHECK.json').read_text());dr={tuple(r['key']):r for r in dense['rows']};group=json.loads((O/'GROUPED_REMOTE_CHECK.json').read_text());original=json.loads((O/'DENSE_ORIGINAL_SOURCE_CHECK.json').read_text());rows=[]
for s in json.loads((O/'SCOPE.json').read_text()):
 if s['worker'] not in ['transformer_dense','transformer_attention_misc']:continue
 key=tuple(s[k] for k in ['baseline','operator','case_id']);p=P/s['worker'];cp=p/s['result']['checker_receipt'];c=json.loads(cp.read_text());r=dr[key] if s['worker']=='transformer_dense' else group;issues=list(r['issues']);v=c['verification']
 if not(c['status']=='verified' and (c['library'],c['op'],c['case_id'])==key and v['checked_outputs']==v['total_outputs']>0 and c['inputs_preserved']):issues.append('Checker incomplete')
 if ref(cp)['sha256']!=r['checker_sha256']:issues.append('Snapshot remote mismatch')
 if r['unknown']!='prior':issues.append('Prior route not preserved')
 if s['worker']=='transformer_dense':
  suffix='post' if key[1]=='MoEPostPermuteFwdOp' else 'quant';gp=p/s['result']['guard_receipt'];exitpath=gp.with_suffix('.exit');rc=int(exitpath.read_text());meta=cp.parent/'guard_snapshot.json';gm=json.loads(meta.read_text());gl=gp.read_text()
  if rc!=0 or gm['owner']!='baseline700-20261026/transformer_dense' or gm['phase']!='running' or gm['acquired_at']>c['started_at']:issues.append('Guard ownership/status mismatch')
  if gm['queue_ticket'] not in gl or key[2] not in gl or '"status": "verified"' not in gl:issues.append('Guard log attribution incomplete')
  launches=[e for e in r['native_evidence'] if 'StreamLaunchKernelV' in e['text']]
  if not launches:issues.append('No actual native launch')
  for e in launches:
   mt=re.search(r'(2026-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',e['text']);when=dt(mt[1]+'T'+mt[2]+'.'+mt[3]+mt[4]+'+00:00')
   if not dt(c['started_at'])<=when<=dt(c['finished_at']):issues.append('Native launch outside checker window')
  sr=next(x for x in original if tuple(x['key'])==key)
  if sr['different'] or sr['missing'] or sr['matched']==0:issues.append('Original source identity incomplete')
  guard=dict(ref(gp),returncode=rc,exit=ref(exitpath),metadata=ref(meta),time_evidence='Acquired timestamp precedes checker; same queue ticket and complete exact key in guard log followed by exit0. No fabricated end timestamp.')
  prov={'remote_identity':ref(O/'DENSE_REMOTE_CHECK.json'),'original_source_check':ref(O/'DENSE_ORIGINAL_SOURCE_CHECK.json'),'original_kernel_files_matched':sr['matched'],'native_evidence':r['native_evidence'],'kernel_hashes':r['kernel_hashes'],'primary_hashes':r['primary_hashes']};recipe=ref(p/('run_post.py' if suffix=='post' else 'run_quant.py'));provider=dense['provider']
 else:
  gp=cp.parent/'guard.json';g=json.loads(gp.read_text());meta=cp.parent/'started.json';gm=json.loads(meta.read_text())['guard'];command=cp.parent/'command.json';cmd=json.loads(command.read_text())
  if not(g['returncode']==0 and g['start']<=dt(c['started_at']).timestamp()<=dt(c['finished_at']).timestamp()<=g['end'] and gm['owner']=='baseline700-20261026/transformer_attention_misc' and cmd[-3:]==[key[2],key[1],key[0]]):issues.append('Guard time/key/owner mismatch')
  guard=dict(ref(gp),returncode=g['returncode'],metadata=ref(meta),command=ref(command),log=ref(cp.parent/'guard.log'))
  orig=json.loads((O/'GROUPED_ORIGINAL_SOURCE_CHECK.json').read_text())
  if not all(x['matches'] for x in orig):issues.append('Original source mismatch')
  prov={'remote_identity':ref(O/'GROUPED_REMOTE_CHECK.json'),'original_source_check':ref(O/'GROUPED_ORIGINAL_SOURCE_CHECK.json'),'compile_receipt':ref(p/'GROUPED_BRIDGE_COMPILE.json'),'bridge':ref(p/'grouped_bridge.cpp'),'review':'Bridge includes unedited original cpp and invokes ParseConfig/LoadGroupList/InferMultiXShapes/IsValidConfig/RunCase. Only process init/reset/finalize are owned by caller; complete original RunCase and kernel retained. Original CSV block/core settings retained. Tested .so hash matches live.','primary_hashes':r['hashes']};recipe=ref(p/'run_dense.py');provider=r['provider']
 rows.append({'key':list(key),'accept_recommended':not issues,'decision':'recommend_accept' if not issues else 'needs_evidence','scope_reasons':s['classification']['reason'],'checker':ref(cp),'verification':v,'inputs_preserved':True,'seed':c['input_seed'],'checker_window':[c['started_at'],c['finished_at']],'guard':guard,'formal_factory':s['result']['factory'],'formal_module':r['module'],'provider':provider,'cpu_stub_dispatch':r['dispatch'],'source_provenance':prov,'runtime_recipe':recipe,'issues':issues})
write('REMAINING_REVIEW.json',{'rows':rows});print([(r['key'],r['issues']) for r in rows])
allrows=[]
for f in ['NN_REVIEW.json','GDN_REVIEW.json','TLA_REVIEW.json','REMAINING_REVIEW.json']:allrows+=json.loads((O/f).read_text())['rows']
for r in allrows:
 cp=pathlib.Path(r['checker']['path']);c=json.loads(cp.read_text());expected=int.from_bytes(hashlib.sha256((c['op']+'::'+c['case_id']).encode()).digest()[:4],'big')
 if c['input_seed']!=expected:r['issues'].append('Canonical seed mismatch');r['accept_recommended']=False;r['decision']='needs_evidence'
 r['canonical_seed_verified']=c['input_seed']==expected
 if r['key'][0]=='ops-nn':r['source_provenance']['targeted_original_and_runtime_identity']=ref(O/'NN_SOURCE_CHECK.json')
write('RECOMMENDED_ACCEPT.json',{'rows':allrows,'partial':False,'scope_count':35})
