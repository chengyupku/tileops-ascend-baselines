import pathlib,json,hashlib,datetime,fcntl,subprocess,sys,importlib,types
sys.dont_write_bytecode=True
D=pathlib.Path(__file__).resolve().parent;W=D.parents[1];A=D/'acceptance_audit02';rows=json.loads((A/'NN_REVIEW.json').read_text())['rows']+json.loads((A/'GDN_REVIEW.json').read_text())['rows'];assert len(rows)==27
(D/'PARENT_AUDIT_1418_SNAPSHOT.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False));cache={}
def sha(p):
 p=str(p)
 if p not in cache:cache[p]=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 return cache[p]
def refs(x):
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x and str(x['path']).startswith('/root/dyq/workspace'):
   assert sha(x['path'])==x['sha256'],x['path']
  for v in x.values():refs(v)
 elif isinstance(x,list):
  for v in x:refs(v)
for r in rows:
 assert r['accept_recommended'] and not r['issues'];refs(r);c=json.loads(pathlib.Path(r['checker']['path']).read_text());assert c['status']=='verified' and c['inputs_preserved'] and not c['oracle_source_changed'];v=c['verification'];assert v['checked_outputs']==v['total_outputs']>0
 if r['key'][0]=='ops-nn':
  s=r['source_provenance'];assert s['launches']
  for p,h in s['kernel_hashes'].items():assert sha(p)==h
  for p,h in s['log_hashes'].items():assert sha(p)==h
  tested=[(p,h) for p,h in c['source_sha256'].items() if p.endswith('/adapter.py') and 'nn_recovery01/' in p];assert len(tested)==1 and tested[0][1]==r['frozen_adapter']['sha256']
  assert c['adapter_sha256']==sha(W/'TileOPs/benchmarks/ascend_baselines.py')
assert json.loads((A/'NN_REVIEW.json').read_text())['previous_provider_prefix_unchanged']
sys.path.insert(0,str(W/'TileOPs'));from benchmarks.ascend_all_pairs import ops_nn
from benchmarks.ascend_all_pairs.ops_nn import integration700_nn_registration01 as m
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);old=m._adapter_for;m._adapter_for=lambda op:types.SimpleNamespace(make=lambda case:('expected',op,case.id))
 for r in rows:
  if r['key'][0]!='ops-nn':continue
  op,cid=r['key'][1:];c=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
  assert ops_nn.PROVIDERS[op](c)==('expected',op,cid)
 m._adapter_for=old
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
# Check live actual server2 provider dispatch for new9, using CPU stub only.
gdn=[r for r in rows if r['key'][0]=='ops-transformer']
code='''import pathlib,sys,importlib,types,hashlib,fcntl,json
sys.dont_write_bytecode=True;sys.path.insert(0,'/opt/tileops-all-pairs/workspace/TileOPs')
rows=ROWS
with open('/tmp/baseline700-provider-ops_transformer.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 from benchmarks.ascend_all_pairs import ops_transformer as p
 from benchmarks.ascend_all_pairs.ops_transformer import integration700_transformer_linear_gdn as m
 h=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest();m.build=lambda case:('expected',case.id)
 for r in rows:
  assert h==r['formal_module']['sha256'];c=types.SimpleNamespace(id=r['key'][2]);assert p.PROVIDERS['GDNFwdOp'](c)==('expected',c.id)
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 print(json.dumps(dict(checked_keys=[r['key'] for r in rows],module_sha256=h,device_imported=False)))
'''.replace('rows=ROWS','rows='+repr(gdn))
r=subprocess.run(['/root/dyq/workspace/.cache/all-pairs/experiments/prompt_efficiency_20261023T1612/remote_guard_repair/bin/ssh','openlibing-DevEnv-055409','python3 -S -'],input=code,text=True,capture_output=True,timeout=40);assert r.returncode==0,r.stderr;(D/'PARENT_AUDIT_1418_GDN_ROUTES.json').write_text(r.stdout)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();out=D/'ACCEPTED.json'
with open(str(out)+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads(out.read_text());by={tuple(x['key']):x for x in d['rows']};before=len(by)
 for r in rows:by[tuple(r['key'])]=dict(r,at=now,accepted=True,parent_review=str(D/'PARENT_AUDIT_1418_REVIEW.json'))
 d.update(rows=list(by.values()),updated_at=now);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(out)
review=dict(at=now,accepted_added=len(by)-before,accepted_total=len(by),nn18_metadata_alert_resolved=True,nn18_actual_native_launch_and_frozen_adapter_checked=True,gdn9_same_original_runtime=True,live_exact_route_checks=27,device_runs=0,receipt_hashes_checked=len(cache),source_review='NN InplaceAddRmsNorm/AddLayerNorm/WeightQuantBatchMatmulV2/QuantBatchMatmulV3 whole original API; lossless input/packed metadata/inplace privateclone binding; full outputs verified. GDN same previously accepted frozen fulloriginal ChunkGatedDeltaRule binding/runtime. Auditors proof and source reviewed; generic adapter fingerprint is framework file not candidate.');(D/'PARENT_AUDIT_1418_REVIEW.json').write_text(json.dumps(review,indent=2));print(json.dumps(review))
