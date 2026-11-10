import pathlib,json,hashlib,re,datetime
O=pathlib.Path(__file__).parent;B=O.parent;ROOT=pathlib.Path('/root/dyq/workspace');W='/opt/tileops-all-pairs/workspace';E=json.load(open(O/'EVIDENCE_INDEX.json'));S=json.load(open(O/'SCOPE.json'));H=json.load(open(O/'server2_hashes.json'));T=json.load(open(O/'server2_static.json'));N=json.load(open(O/'NATIVE_LOG_AUDIT.json'));M=json.load(open(O/'ATTENTION_MANIFEST_AUDIT.json'));rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(s):return datetime.datetime.fromisoformat(s).timestamp()
assert all(v['sha256']==v['expected'] for v in M['selected_pins'].values())
for r,x in zip(S,E):
 if r['worker']!='transformer_attention_misc':continue
 c=x['checker_summary'];g=json.load(open(x['guard']['path']));gp=pathlib.Path(x['guard']['path']);cmd=json.load(open(gp.parent/'command.json'));assert cmd[:2]==['npu-guard','run'];assert 'baseline700-20261026/transformer_attention_misc' in cmd;assert g['returncode']==0 and g['start']<stamp(c['started_at'])<stamp(c['finished_at'])<g['end'];cid=r['case_id'];op=r['operator']
 if cmd[-1].endswith('_batch.json') or cmd[-1]=='mha_batch.json':assert [cid,op,r['baseline']] in json.load(open(B/r['worker']/cmd[-1]))
 else:assert cid in cmd
 assert c['inputs_preserved'] and x['inputs_metadata_equal'] and not c['oracle_source_changed'] and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 for typ in ['checker','guard']:assert H[x[typ]['path'].replace(str(ROOT),W)]['sha256']==x[typ]['sha256']
 group= ('attention_gqa128' if '128k' in cid else 'attention_gqa64') if op=='GQADenseFwdOp' else ('mha_b2' if cid=='serving-b2-float16' else 'mha_batch') if op.startswith('MHA') else ('mla_b1' if cid=='mla-tp-b1-float16' else 'mla_batch')
 evidence=[];kernels=set()
 for f in N[group]:
  for hit in f['hits']:
   text=hit['text'];m=re.search(r'2026-10-26-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',text)
   if not m:continue
   t='2026-10-26T'+m[1]+'.'+m[2]+m[3]+'+00:00'
   if c['started_at']<=t<=c['finished_at']:
    if 'StreamLaunchKernelV1:' in text:
     kernels.add(re.search(r'kernel_name=([^, ]+)',text)[1])
    evidence.append({'path':f['path'],'sha256':f['sha256'],'line':hit['line'],'text':text})
 assert kernels,(cid,'no exact interval launch')
 for kernel in kernels:assert any('LogTaskDetails' in h['text'] and 'kernel['+kernel+']' in h['text'] and '/vendors/classicpilot_attention_transformer/' in h['text'] for f in N[group] for h in f['hits'])
 factory=r['result']['factory'];module=factory.split('.')[0] if not factory.startswith('benchmarks.') else factory.split('.')[-2];mp=W+'/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/'+module+'.py';mh=H[mp]['sha256'];assert sha(B/r['worker']/(module+'.py'))==mh
 assert cid in T[W+'/TileOPs/benchmarks/ascend_all_pairs/ops_transformer/__init__.py']['text']
 if op!='GQADenseFwdOp':assert mh in x['source'].values()
 rows.append({'key':x['key'],'accept_recommended':True,'scope_reasons':r['classification'],'checker':x['checker'],'guard':x['guard'],'guard_returncode':0,'batch_command':{'path':str(gp.parent/'command.json'),'sha256':sha(gp.parent/'command.json')},'formal_factory':factory,'module_sha256':mh,'source_provenance':{'manifest':M['manifest_path'],'manifest_sha256':M['manifest_sha256'],'primary_pins_audit':str(O/'ATTENTION_MANIFEST_AUDIT.json'),'native_evidence':evidence},'review':'Whole original FlashAttentionScore graph call; original vendor dynamic compile name matched to exact checker-interval device launches, including shared first-case process logs. GQA BSND binding; MHA lossless page gather and independent batch calls; MLA concatenated Q/K features and original full attention scale. Full output checked with unchanged canonical CPU oracle, seed and tolerance; exact router preserves previous provider.','issues':[]})
def put(n,j):
 p=O/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
put('ATTENTION_RECOMMENDED_ACCEPT.json',{'rows':rows});a=json.load(open(O/'RECOMMENDED_ACCEPT.json'))['rows'];put('RECOMMENDED_ACCEPT.json',{'rows':a+rows});print('attention',len(rows))
