import pathlib,json,hashlib,ast,datetime
O=pathlib.Path(__file__).parent;B=O.parent;R=pathlib.Path('/root/dyq/workspace');W='/opt/tileops-all-pairs/workspace';E=json.load(open(O/'EVIDENCE_INDEX.json'));S=json.load(open(O/'SCOPE.json'));H=json.load(open(O/'server1_hashes.json'));X=json.load(open(O/'server1_extra_hashes.json'));T=json.load(open(O/'server1_static.json'));rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(s):return datetime.datetime.fromisoformat(s).timestamp()
for r,x in zip(S,E):
 if r['worker']!='tla_simple_tilelang':continue
 c=x['checker_summary'];p=pathlib.Path(x['checker']['path']);g=json.load(open(x['guard']['path']));meta=json.load(open(p.parent/'GUARD_RUNNING.json'));assert c['inputs_preserved'] and x['inputs_metadata_equal'] and not c['oracle_source_changed'];assert c['verification']['checked_outputs']==c['verification']['total_outputs']==1;assert g['started']<=stamp(c['started_at'])<stamp(c['finished_at'])<=g['finished'];assert meta['owner']=='baseline700-20261026/tla_simple_tilelang'
 for typ in ['checker','guard']:assert H[x[typ]['path'].replace(str(R),W)]['sha256']==x[typ]['sha256']
 f=r['result']['factory'].split(':')[0];mp=R/'TileOPs'/('/'.join(f.split('.'))+'.py');h=sha(mp) if mp.exists() else sha(B/r['worker']/(f.split('.')[-1]+'.py'));assert H[W+'/TileOPs/'+('/'.join(f.split('.'))+'.py')]['sha256']==h
 tree=ast.parse((B/r['worker']/(f.split('.')[-1]+'.py')).read_text());config=ast.literal_eval(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='config').body[0].value);assert [r['operator'],r['case_id']] in config['keys']
 for pin,expected in config['pins'].items():assert T[pin]['sha256']==expected
 adapterpin=next(v for k,v in config['pins'].items() if k.endswith('/adapter.py'));assert adapterpin in x['source'].values()
 for q,v in x['source'].items():
  q=q if q.startswith('/') else W+'/'+q
  assert X[q].get('sha256')==v,(q,X[q],v)
 calls=list((p.parent/'jit/launch-evidence').glob('*/calls-*.json'));assert calls
 for cp in calls:
  cj=json.load(open(cp));assert cj['successful_host_launches']>0 and cj['pid']==meta['child']['pid']
  for call in cj['calls']:
   sp=R/call['generated_source']['path'];assert sha(sp)==call['generated_source']['sha256']
 if '_v3' in str(p):
  gp=B/r['worker']/'select_batch_v3.guard.log';ep=B/r['worker']/'select_batch_v3.exit';assert ep.read_text().strip()=='0' and meta['queue_ticket'] in gp.read_text();batch={'log':str(gp),'sha256':sha(gp),'exit':str(ep),'returncode':0,'running_metadata':str(p.parent/'GUARD_RUNNING.json')}
 else:
  gp=B/r['worker']/('element_split_probe.guard.json' if '_v5' in str(p) else 'element_remaining_bounded.guard.json');gj=json.load(open(gp));assert gj['returncode']==0 and [r['operator'],r['case_id']] in gj['keys'];assert gj['started']<stamp(c['started_at'])<stamp(c['finished_at'])<gj['finished'];batch={'path':str(gp),'sha256':sha(gp),'returncode':0}
 rows.append({'key':x['key'],'accept_recommended':True,'scope_reasons':r['classification'],'checker':x['checker'],'guard':x['guard'],'actual_batch_guard':batch,'formal_factory':r['result']['factory'],'module_sha256':h,'source_provenance':{'frozen_pins':config['pins'],'launch_evidence':[{'path':str(z),'sha256':sha(z)} for z in calls]},'review':'Original AST definitions and upstream tile/pass config; select mask packing/FP32 widening; bitwise int32 reinterpretation into int16 lanes; bool XOR1. Neg/Clamp split independent rows into four full original API calls, concatenate only, no cross-row reduction/new math. Other cases single original call. All checker outputs and inputs preserved, exact registered config and old routes retained. Remote source/runtime/generated/binary evidence hashes verified.','issues':[]})
def put(n,j):
 p=O/n;q=p.with_suffix('.tmp');q.write_text(json.dumps(j,indent=2,ensure_ascii=False));q.replace(p)
put('TLA_RECOMMENDED_ACCEPT.json',{'rows':rows});a=json.load(open(O/'RECOMMENDED_ACCEPT.json'))['rows'];put('RECOMMENDED_ACCEPT.json',{'rows':a+rows});print('TLA',len(rows))
