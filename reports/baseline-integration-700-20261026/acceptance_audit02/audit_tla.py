import json,pathlib,hashlib,datetime
W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026';O=P/'acceptance_audit02';T=P/'tla_simple_tilelang'
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}
def write(n,x):
 t=O/(n+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(O/n)
remote=json.loads((O/'TLA_REMOTE_CHECK.json').read_text());rr={tuple(r['key']):r for r in remote['rows']};rows=[]
for s in json.loads((O/'SCOPE.json').read_text()):
 if s['worker']!='tla_simple_tilelang':continue
 key=tuple(s[k] for k in ['baseline','operator','case_id']);r=rr[key];cp=pathlib.Path(s['result']['checker_receipt']);c=json.loads(cp.read_text());cg=cp.parent/'guard.json';child=json.loads(cg.read_text());mp=cp.parent/'GUARD_RUNNING.json';meta=json.loads(mp.read_text());batch='pto_distinct_probe' if key[1]=='SqrtFwdOp' else 'isolated_probe';gp=T/(batch+'.guard.json');g=json.loads(gp.read_text());log=T/(batch+'.guard.log');issues=list(r['issues']);v=c['verification'];start=datetime.datetime.fromisoformat(c['started_at']).timestamp();end=datetime.datetime.fromisoformat(c['finished_at']).timestamp()
 if not(c['status']=='verified' and c['case_id']==key[2] and v['checked_outputs']==v['total_outputs']>0 and c['inputs_preserved']):issues.append('Checker incomplete')
 if not(g['started']<=start<=end<=g['finished'] and child['returncode']==0 and child['finished']>=end and list(key[1:]) in g['keys']):issues.append('Batch containment/key mismatch')
 req=datetime.datetime.fromisoformat(meta['requested_at']).timestamp()
 if abs(req-g['started'])>1 or meta['owner']!='baseline700-20261026/tla_simple_tilelang':issues.append('Reservation identity mismatch')
 if g['returncode']==76:issues.append('External interference invalidates batch')
 if 'acquired device 0 for baseline700-20261026/tla_simple_tilelang' not in log.read_text():issues.append('Missing acquired guard')
 if ref(cp)['sha256']!=r['checker_sha256']:issues.append('Remote checker differs')
 if not r['calls'] or not all(x['successful_host_launches']>0 for x in r['calls']):issues.append('No launched original JIT kernel')
 rows.append({'key':list(key),'accept_recommended':not issues,'decision':'recommend_accept' if not issues else 'needs_evidence','scope_reasons':s['classification']['reason'],'checker':ref(cp),'verification':v,'inputs_preserved':True,'seed':c['input_seed'],'checker_window':[c['started_at'],c['finished_at']],'guard':dict(ref(gp),returncode=g['returncode'],child_receipt=ref(cg),child_returncode=child['returncode'],metadata=ref(mp),log=ref(log)),'guard_qualification':'Batch ended later with '+str(g['returncode'])+'; exact case completed full checker and synchronization with child receipt 0 beforehand. Not a clean-exit-0 batch; no 76 interference evidence.','formal_factory':s['result']['factory'],'formal_module':r['module'],'provider':remote['provider'],'source_provenance':{'remote_hash_check':ref(O/'TLA_REMOTE_CHECK.json'),'launch_calls':r['calls'],'original_api_review':'Full original sqrt / matmul / batch_matmul. Upstream blocks retained; only lossless dtype widening, layout/pad/crop and independent expert mapping per SCOPE. Frozen adapter hashes equal tested source.','runtime':'Original pinned TLA runtime and PTO backend; all checker source/runtime/JIT binary hashes rechecked remotely.'},'cpu_stub_dispatch':r['dispatch'],'issues':issues})
write('TLA_REVIEW.json',{'rows':rows});prior=[]
for f in ['NN_REVIEW.json','GDN_REVIEW.json']:prior+=json.loads((O/f).read_text())['rows']
write('RECOMMENDED_ACCEPT.json',{'rows':prior+rows,'partial':True})
write('ALERTS.json',{'alerts':[{'severity':'accounting_correction','blocking':False,'keys':[r['key'] for r in rows],'finding':'TLA per-case guard.json returncode 0 is not the enclosing npu-guard returncode. Sqrt batch=139, Gemm/Bmm/MoE batch=124. Preserve true batch exit; per-case complete successes predate later failure.','evidence':[ref(T/'pto_distinct_probe.guard.json'),ref(T/'isolated_probe.guard.json'),ref(T/'run_isolated_batch.py'),ref(T/'run_batch9.py')]}]})
print([(r['key'][1:],r['issues'],r['guard']['returncode']) for r in rows])
