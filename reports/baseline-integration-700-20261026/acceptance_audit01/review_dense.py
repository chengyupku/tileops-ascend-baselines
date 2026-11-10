import pathlib,json,hashlib,re,datetime
O=pathlib.Path(__file__).parent;B=O.parent;ROOT=pathlib.Path('/root/dyq/workspace');W='/opt/tileops-all-pairs/workspace';E=json.load(open(O/'EVIDENCE_INDEX.json'));S=json.load(open(O/'SCOPE.json'));H=json.load(open(O/'server1_hashes.json'));X=json.load(open(O/'server1_extra_hashes.json'));T=json.load(open(O/'server1_static.json'));N=json.load(open(O/'NATIVE_LOG_AUDIT.json'));rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for r,x in zip(S,E):
 if r['worker']!='transformer_dense':continue
 c=x['checker_summary'];op=r['operator'];cid=r['case_id'];issues=[];p=pathlib.Path(x['checker']['path']);p=p if p.is_absolute() else ROOT/p;g=pathlib.Path(x['guard']['path']);g=g if g.is_absolute() else ROOT/g;ep=g.with_suffix('.exit');rc=int(ep.read_text());assert rc in [0,1];sp=p.parent/'guard_snapshot.json';meta=json.load(open(sp));assert meta['owner']=='baseline700-20261026/transformer_dense' and meta['acquired_at']<c['started_at'];assert meta['queue_ticket'] in g.read_text();assert c['inputs_preserved'] and x['inputs_metadata_equal'] and not c['oracle_source_changed'] and c['verification']['checked_outputs']==c['verification']['total_outputs']>0
 module=r['result']['factory'].split('.')[-2];mp=W+'/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'+module+'.py';mh=H[mp]['sha256'];assert mh==sha(B/r['worker']/(module+'.py'));assert cid in T[W+'/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py']['text']
 assert H[str(p).replace(str(ROOT),W)]['sha256']==x['checker']['sha256'];assert H[str(g).replace(str(ROOT),W)]['sha256']==x['guard']['sha256']
 if op=='YaRNFwdOp':
  issues=['Tested rope module 70e559... differs from live ccd667...; exact old source/diff needed to establish active YaRN branch equivalence.','No original device-launch attribution located in this low-log-level successful YaRN run; kernel cache alone does not establish source and launch.','Guard exit is genuinely 1 from subsequent LongRoPE failure; prior full YaRN checker can stand on its own if other evidence is completed.']
 elif op=='FusedTopKFwdOp':
  proof=json.load(open(O/'TOPK_IMPORT_RELOCATION_PROOF.json'));assert proof['formal_sha256']==mh and proof['tested_sha256'] in x['source'].values()
 else:assert mh in x['source'].values(), (op,module,mh)
 evidence=[]
 if op in ['LongRoPEFwdOp','FusedTopKFwdOp']:
  group='longrope' if op=='LongRoPEFwdOp' else 'topk';kernels=set()
  for f in N[group]:
   for hit in f['hits']:
    text=hit['text'];m=re.search(r'2026-10-26-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',text)
    if not m:continue
    t='2026-10-26T'+m[1]+'.'+m[2]+m[3]+'+00:00'
    if c['started_at']<=t<=c['finished_at']:
     evidence.append({'path':f['path'],'sha256':f['sha256'],'line':hit['line'],'text':text})
     if 'StreamLaunchKernelV1:' in text:kernels.add(re.search(r'kernel_name=([^, ]+)',text)[1])
  assert kernels
  for k in kernels:assert any('LogTaskDetails' in z['text'] and 'kernel['+k.split('__kernel')[0]+']' in z['text'] for z in evidence),(op,k,[z['text'][:200] for z in evidence])
 if op=='GeluTanhAndMulFwdOp':
  for q,h in x['source'].items():
   rq=q if q.startswith('/') else W+'/'+q
   actual=(X|H)[rq]['sha256'];assert actual==h
  assert sha(B/'transformer_dense/BIAS_GATE_GELU.cpp')==sha(ROOT/'ascend_baselines/ops-transformer/experimental/moe/biasgategelu/BIAS_GATE_GELU.cpp')
  evidence=[{'path':str(B/'transformer_dense/BUILD.json'),'sha256':sha(B/'transformer_dense/BUILD.json')},{'source_path':str(B/'transformer_dense/BIAS_GATE_GELU.cpp'),'source_sha256':sha(B/'transformer_dense/BIAS_GATE_GELU.cpp'),'launch':'Original source lines 172/175/198 bind PrivateUse1 schema directly to biasGateGelu_kernel<<<blockDim>>>; exact loaded SO hash matches checker and current remote. No ACLNN/CANN dispatcher fallback.'}]
 rows.append({'key':x['key'],'accept_recommended':not issues,'disposition':'needs_evidence' if issues else 'recommend_accept','scope_reasons':r['classification'],'checker':x['checker'],'guard':x['guard'],'actual_guard_exit':{'path':str(ep),'sha256':sha(ep),'returncode':rc},'guard_running':{'path':str(sp),'sha256':sha(sp)},'formal_factory':r['result']['factory'],'module_sha256':mh,'source_provenance':{'kernel_hashes':str(O/'DENSE_ORIGINAL_KERNEL_HASHES.json'),'native_evidence':evidence},'review':'Full canonical checker and preserved inputs; original complete API and standard dtype/state bindings. TopK large raw index max_abs_err is not a failed tolerance: unchanged custom oracle validates unique legal expert selection and associated weights independent of ordering (workloads/moe.py:34-98). GELU zero bias keeps original block40 and original source/binary. Graph TopK/LongRoPE compile source and exact interval device launch established.','issues':issues,'minimum_completion':'Provide old tested YaRN source plus branch diff/current-module existing checker and exact original-kernel attribution; reuse receipts, no rerun requested.' if issues else None})
def put(n,j):
 p=O/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
put('DENSE_RECOMMENDED_ACCEPT.json',{'rows':rows});a=json.load(open(O/'RECOMMENDED_ACCEPT.json'))['rows'];put('RECOMMENDED_ACCEPT.json',{'rows':a+rows});print('dense',len(rows),'accept',sum(r['accept_recommended'] for r in rows))
