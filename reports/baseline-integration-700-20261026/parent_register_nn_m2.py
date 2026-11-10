import pathlib,json,hashlib,fcntl,sys,types,os,ast,sqlite3,datetime
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_m2_source_audit1';N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
db=sqlite3.connect('file:/root/dyq/codex-workers/workers.sqlite?mode=ro',uri=True);assert db.execute('select status from jobs where id=?',('a86986266d2041a2',)).fetchone()[0]=='completed'
r=read(P/'PARENT_REVIEW.json');assert r['status']=='eligible_pending_formal_registration_after_active_nn_checks'
packet=read(P/'PACKET_HASHES.json');print('packet format',type(packet).__name__)
# Existing parent review independently reconciled historical source race. Recheck private registration bytes.
assert sha(P/'integration700_nn_m2_source_audit1.py')==read(P/'CPU_PROOF.json')['module_sha256']
patch=(P/'REGISTRATION.patch').read_text();append='\n'.join(x[1:] for x in patch.splitlines() if x.startswith('+') and not x.startswith('+++'))+'\n'
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 old=(N/'__init__.py').read_text();assert '_nn_m2_route' not in old;assert '_nn_dense_resume8_route' in old;ast.parse(old+append)
 (P/'PARENT_PROVIDER_BEFORE.py').write_text(old)
 tmp=N/'__init__.py.parent.tmp';tmp.write_text(old+append);os.replace(tmp,N/'__init__.py')
 try:
  from benchmarks.ascend_all_pairs import ops_nn,registry
  m=ops_nn._nn_m2;assert sha(pathlib.Path(m._load().__file__))==m._SHA256
  orig=m._load;m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id))
  os.environ['OPS_NN_VENDOR']=str(m._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._VENDOR)
  case=types.SimpleNamespace(id=r['key'][2],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GemmFwdOp')))
  assert registry.candidate_factory(case,'ops-nn')()==('expected',case.id)
  case.id='__unknown__';assert ops_nn._nn_m2_route(case,_prior=lambda c:'prior')=='prior';m._load=orig
  n=ops_nn._nn_dense_resume8;n._load=lambda:types.SimpleNamespace(make=lambda c:('nn8',c.id))
  for op,cid in n._KEYS:
   case.id=cid;assert registry.candidate_factory(case,'ops-nn')()==('nn8',cid)
  assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
 receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registered_key':r['key'],'old_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_sha256':sha(N/'__init__.py'),'module_sha256':sha(P/'integration700_nn_m2_source_audit1.py'),'actual_disk_registry_verified':True,'NN8_routes_preserved':8,'torch_imported':False,'NPU_runs':0,'lock':'/tmp/baseline700-provider-ops_nn.lock'}
 (P/'PARENT_REGISTRATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
