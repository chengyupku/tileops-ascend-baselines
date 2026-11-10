import pathlib,json,hashlib,re,datetime,sys,os,types,fcntl
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_dense_resume8';N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';read=lambda p:json.loads(p.read_text());cache={}
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
rows=read(P/'RESULT.json')['rows'];assert len(rows)==8
scope={(r['baseline'],r['operator'],r['case_id']) for r in read(P/'SCOPE.json')};assert scope=={(r['baseline'],r['operator'],r['case_id']) for r in rows}
refs={};kernels=[]
for r in rows:
 out=pathlib.Path(r['checker']).parent;c=read(out/'checker.json');f=read(out/'FROZEN.json');g=read(out/'GUARD_RESULT.json');gm=read(out/'GUARD_METADATA.json')
 assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1
 assert c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed'] and c['independent_cpu_reference']
 assert c['input_seed']==int.from_bytes(hashlib.sha256(('GemmFwdOp::'+c['case_id']).encode()).digest()[:4],'big')
 assert g['returncode']==0 and gm['owner']=='baseline700-20261026/nn_dense_resume8' and gm['acquired_at']<c['started_at']
 assert read(out/'COMPLETION.json')['checker_returncode']==0
 for p,h in f['receipts'].items():assert sha(out/p)==h,p
 for p,h in c['source_sha256'].items():assert sha(W/p)==h,p
 for p,h in c['oracle_source_sha256'].items():assert sha(P/'snapshot'/p)==h and sha(W/p)==h,p
 ks=set()
 for e in f['native_compiles']:
  assert 'classicpilot_nn_impl.dynamic.mat_mul_v3' in e['text'] and 'result[success]' in e['text'];ks.add(re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0])
 for e in f['native_compiles']+f['native_launches']:
  kernel=re.search(r'te_matmulv3_[a-f0-9]+',e['text'])[0];assert kernel in ks
  stamp=re.search(r'2026-10-26-\d\d:\d\d:\d\d\.\d\d\d\.\d\d\d',e['text'])[0];ts=stamp[:10]+'T'+stamp[11:19]+'.'+stamp[20:23]+stamp[24:27]+'+00:00'
  assert c['started_at']<=ts<=c['finished_at'];refs.setdefault(e['file'],{})[e['line']]=e['text']
 assert f['native_compiles'] and f['native_launches'] and f['binary_sha256']
 for p,h in f['binary_sha256'].items():
  assert sha(p)==h;meta=read(pathlib.Path(p).with_suffix('.json'));assert meta['sha256']==h;assert pathlib.Path(p).stem in ks
 kernels.extend(ks)
for p,ls in refs.items():
 lines=pathlib.Path(p).read_text(errors='replace').splitlines()
 for n,t in ls.items():assert lines[n-1]==t
comp=read(D/'nn_recovery02/SOURCE_COMPARE.json')
for x in comp:assert x['identical'] and sha(x['source'])==sha(x['build'])==x['sha256']
assert sha(P/'adapter.py')=='037f902f6779aa1d84dad1d5d04777e6754cec5e9b3d653b7339ac5f0dd200d7'
reg=read(P/'REGISTRATION.json');assert sha(N/'integration700_nn_dense_resume8.py')==reg['module_sha256']==sha(P/'integration700_nn_dense_resume8.py');assert sha(N/'__init__.py')==reg['new_sha256'];assert (N/'__init__.py').read_text().startswith((P/'provider_before.py').read_text())
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 from benchmarks.ascend_all_pairs import ops_nn,registry
 m=ops_nn._nn_dense_resume8;assert sha(m._load().__file__)==sha(P/'adapter.py');m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id))
 os.environ['OPS_NN_VENDOR']=str(m._VENDOR);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(m._VENDOR)
 for _,op,cid in scope:
  case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)));assert registry.candidate_factory(case,'ops-nn')()==('expected',cid)
 case.id='__unknown__';assert ops_nn._nn_dense_resume8_route(case,_prior=lambda c:'prior')=='prior'
 assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
review={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'eligible_pending_ledger_commit','keys':sorted(scope),'hashes_checked':len(cache),'native_raw_lines_checked':sum(map(len,refs.values())),'compiled_kernels':kernels,'retained_ELF_rehashed':8,'original_source_build_pairs_checked':len(comp),'formal_registry_exact_keys':8,'device_runs':0,'successful_numeric_reruns':0,'boundary':'Unchanged original complete MatMulV3 adapter, lossless BF16 to FP32 and canonical layout only, final output cast. Full unchanged reference and canonical tolerance; independent source/oracle/seed and native compile/launch/ELF checks.'}
(P/'PARENT_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps(review))
