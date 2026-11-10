import pathlib,json,hashlib,subprocess,fcntl,datetime,concurrent.futures
D=pathlib.Path(__file__).resolve().parent;W=D.parents[1];A=D/'acceptance_audit02'
allrows=json.loads((A/'RECOMMENDED_ACCEPT.json').read_text());allrows=allrows if isinstance(allrows,list) else allrows['rows']
accepted={tuple(r['key']) for r in json.loads((D/'ACCEPTED.json').read_text())['rows']}
rows=[r for r in allrows if tuple(r['key']) not in accepted];assert len(rows)==3
cache={}
def sha(p):
 p=str(p)
 if p not in cache:cache[p]=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 return cache[p]
def checkrefs(x):
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x and x['path'].startswith(str(W)):assert sha(x['path'])==x['sha256'],x['path']
  for v in x.values():checkrefs(v)
 elif isinstance(x,list):
  for v in x:checkrefs(v)
for r in rows:
 assert r['accept_recommended'] and not r['issues'];checkrefs(r)
 c=json.loads(pathlib.Path(r['checker']['path']).read_text());assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'];v=c['verification'];assert v['checked_outputs']==v['total_outputs']>0;assert r['guard']['returncode']==0
 assert c['input_seed']==r['seed']
 if 'exit' in r['guard']:assert pathlib.Path(r['guard']['exit']['path']).read_text().strip()=='0'
 else:assert json.loads(pathlib.Path(r['guard']['path']).read_text())['returncode']==0
code='''import sys,types,importlib,pathlib,hashlib,json,fcntl
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');sys.path.insert(0,str(W/'TileOPs'))
rows=ROWS
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
out=[]
for r in rows:
 provider=r['key'][0].replace('-','_')
 with open('/tmp/baseline700-provider-'+provider+'.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=importlib.import_module('benchmarks.ascend_all_pairs.'+provider)
  mn,fn=r['formal_factory'].rsplit('.',1);m=importlib.import_module(mn)
  assert sha(m.__file__)==r['formal_module']['sha256']
  orig=getattr(m,fn);aliases={k:v for k,v in vars(p).items() if v is orig}
  stub=lambda *a: ('exact',r['key']);setattr(m,fn,stub)
  for k in aliases:setattr(p,k,stub)
  case=types.SimpleNamespace(id=r['key'][2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=r['key'][1])))
  assert p.PROVIDERS[r['key'][1]](case)==('exact',r['key'])
  setattr(m,fn,orig)
  for k,v in aliases.items():setattr(p,k,v)
  prov=r['source_provenance'];checked={}
  for field in ['primary_hashes','kernel_hashes']:
   for path,h in prov.get(field,{}).items():assert sha(path)==h,path;checked[path]=h
  for item in prov.get('native_evidence',[]):
   path=item['file'];assert sha(path)==item['log_sha256'];line=pathlib.Path(path).read_text(errors='replace').splitlines()[item['line']-1];assert line==item['text'];checked[path]=item['log_sha256']
  out.append({'key':r['key'],'module_sha256':sha(m.__file__),'exact_route':True,'critical_hashes':checked})
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
print(json.dumps({'rows':out,'device_imported':False}))
'''
ssh='/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh'
def run(host,subset):
 p=subprocess.run([ssh,host,'python3 -S -'],input=code.replace('rows=ROWS','rows='+repr(subset)),text=True,capture_output=True,timeout=50)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 fs=[ex.submit(run,'openlibing-DevEnv-661495',[r for r in rows if r['key'][0]=='ops-transformer']),ex.submit(run,'openlibing-DevEnv-055409',[r for r in rows if r['key'][0]=='ops-tensor'])]
 results=[f.result() for f in fs]
(D/'PARENT_REMAINING02_LIVE_ROUTE_CHECK.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
assert all(not x['missing'] and not x['different'] and x['matched']>0 for x in json.loads((A/'DENSE_ORIGINAL_SOURCE_CHECK.json').read_text()))
assert all(x['matches'] for x in json.loads((A/'GROUPED_ORIGINAL_SOURCE_CHECK.json').read_text()))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();review_path=D/'PARENT_REMAINING02_REVIEW.json';out=D/'ACCEPTED.json'
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(out.read_text());by={tuple(r['key']):r for r in ledger['rows']};before=len(by)
 for r in rows:by[tuple(r['key'])]=dict(r,accepted=True,at=now,parent_review=str(review_path))
 ledger.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.parent.tmp');tmp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2));tmp.replace(out)
review={'at':now,'accepted_added':len(by)-before,'accepted_total':len(by),'keys':[r['key'] for r in rows],'local_evidence_hashes_checked':len(cache),'live_exact_route_checks':3,'NPU_commands':0,'source_boundary_review':'Read quant/post/grouped adapters and C++ bridge: whole original GroupedMatmulV5, MoeFinalizeRoutingV2 and unchanged ops-tensor ParseConfig/LoadGroupList/InferMultiXShapes/RunCase. Transpose/scale metadata packing/physical prefix and binary IO only, full outputs. Auditor original-source equality and original CSV launch settings confirmed; live critical native logs/kernels/binding/runtime hashes match.','route_evidence':str(D/'PARENT_REMAINING02_LIVE_ROUTE_CHECK.json')}
review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2));print(json.dumps(review,ensure_ascii=False))
