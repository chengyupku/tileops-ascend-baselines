import json,pathlib,hashlib,datetime,re,sys,types
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026';O=P/'acceptance_audit02'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def write(n,x):
 t=O/(n+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(O/n)
def dt(s):return datetime.datetime.fromisoformat(s)
regs=json.loads((P/'nn_registration01/REGISTRATION.json').read_text())['rows'];scope={tuple(r[k] for k in ['baseline','operator','case_id']):r for r in json.loads((O/'SCOPE.json').read_text())}
# Import only audited pure-Python formal module. Stub all runtime dependencies; no torch import.
for name in ['benchmarks','benchmarks.ascend_baselines','benchmarks.ascend_all_pairs','benchmarks.ascend_all_pairs.ops_nn','benchmarks.ascend_all_pairs.ops_nn.graph','benchmarks.ascend_all_pairs.ops_nn.common','torch','workloads','workloads.gemm']:
 sys.modules[name]=types.ModuleType(name)
sys.modules['workloads.gemm'].unrepack_w4a16_weight=lambda x:x
sys.modules['benchmarks.ascend_baselines'].UnsupportedCase=type('UnsupportedCase',(Exception,),{})
sys.modules['benchmarks.ascend_all_pairs.ops_nn.graph'].GraphCall=lambda library:types.SimpleNamespace(codes={},library=library)
sys.modules['benchmarks.ascend_all_pairs.ops_nn.common'].candidate=lambda b,run,api,adapt,directory:dict(api=api,function=run.__name__,library=b.library,directory=directory)
mp=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/integration700_nn_registration01.py'
m=types.ModuleType('benchmarks.ascend_all_pairs.ops_nn.integration700_nn_registration01');exec(compile(mp.read_text(),str(mp),'exec'),m.__dict__);sys.modules[m.__name__]=m
m2=types.ModuleType('benchmarks.ascend_all_pairs.ops_nn.integration700_nn_parent_last3');p2=mp.parent/'integration700_nn_parent_last3.py';exec(compile(p2.read_text(),str(p2),'exec'),m2.__dict__);sys.modules[m2.__name__]=m2
setattr(sys.modules['benchmarks.ascend_all_pairs.ops_nn'],'integration700_nn_parent_last3',m2)
assert not m2._KEYS.intersection(m._KEYS)
ip=mp.parent/'__init__.py';text=ip.read_text();prefix,suffix=text.split('# Exact registration: baseline-integration-700-20261026/nn_registration01',1)
prior=lambda case:'prior-preserved';ns={'__package__':'benchmarks.ascend_all_pairs.ops_nn','PROVIDERS':{op:prior for _,op,_ in scope}}
exec(compile(suffix,str(ip),'exec'),ns)
rows=[]
for r in regs:
 key=tuple(r['key']);s=scope[key];issues=[];cpath=pathlib.Path(r['checker']['path']);c=json.loads(cpath.read_text());gpath=pathlib.Path(r['guard']['path']);g=json.loads(gpath.read_text());ap=pathlib.Path(r['attribution']['path']);a=json.loads(ap.read_text());v=c['verification']
 if (c['library'],c['op'],c['case_id'])!=key or c['status']!='verified' or not(v['checked_outputs']==v['total_outputs']>0) or not c.get('inputs_preserved'):issues.append('Incomplete checker/key/input preservation')
 if not(g['returncode']==0 and dt(g['requested_at'])<=dt(c['started_at'])<=dt(c['finished_at'])<=dt(g['finished_at'])):issues.append('Guard status/time mismatch')
 metadata=cpath.parent/'GUARD_METADATA.json';gm=json.loads(metadata.read_text());command=gpath.parent/'COMMAND.json';cmd=json.loads(command.read_text());batch=pathlib.Path(cmd[-1]);batchrows=json.loads(batch.read_text())
 if not any(x[:3]==list(key) and x[3]==cpath.parent.name for x in batchrows):issues.append('Batch identity mismatch')
 if gm.get('owner')!='baseline700-20261026/nn_recovery01' or gm.get('phase')!='running':issues.append('Guard ownership mismatch')
 guardlog=gpath.parent/'guard.log';gl=guardlog.read_text()
 if 'acquired device 0' not in gl or 'external NPU' in gl:issues.append('Guard log missing acquisition or interference')
 frozen=pathlib.Path(r['adapter']['frozen_path']);tested=pathlib.Path(r['adapter']['tested_case_path']);h=sha(frozen)
 if sha(tested)!=h or c['source_sha256'].get(str(tested.relative_to(W)))!=h:issues.append('Tested adapter not equal frozen')
 hashes={};bad=[]
 for p,hv in c['oracle_source_sha256'].items():
  hashes[p]=sha(W/p)
  if hashes[p]!=hv:bad.append(p)
 for name in ['common.py','graph.py','catalog.py','truth.py','tensorlist.py']:
  p='TileOPs/benchmarks/ascend_all_pairs/ops_nn/'+name;hashes[p]=sha(W/p)
  if hashes[p]!=c['source_sha256'].get(p):bad.append(p)
 if bad:issues.append('Source drift: '+str(bad))
 if sha(W/'TileOPs/scripts/ascend_pair_check.py')!=c['verifier_sha256']:issues.append('Verifier drift')
 # Validate original log records, kernel bytes, and actual device launch within exact checker window.
 launches=[];record_sources={}
 for rec in a['records']:
  if 'StreamLaunchKernelV1:' not in rec['text']:continue
  line=pathlib.Path(rec['file']).read_text(errors='replace').splitlines()[rec['line']-1]
  if not line.startswith(rec['text']):issues.append('Attribution record does not match original log')
  mt=re.search(r'(2026-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',line)
  when=dt(mt[1]+'T'+mt[2]+'.'+mt[3]+mt[4]+'+00:00')
  kernel=re.search(r'kernel_name=([^, ]+)',line)[1]
  if not dt(c['started_at'])<=when<=dt(c['finished_at']):issues.append('Launch outside checker')
  if not any(pathlib.Path(p).stem in kernel for p in a['kernel_hashes']):issues.append('Launch kernel not attributed binary')
  launches.append({'file':rec['file'],'line':rec['line'],'at':when.isoformat(),'kernel':kernel});record_sources[rec['file']]=sha(rec['file'])
 if not launches:issues.append('No actual own kernel launch')
 for p,hv in a['kernel_hashes'].items():
  if sha(p)!=hv:issues.append('Kernel binary drift '+p)
 case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=key[1])),id=key[2],params=c['case_snapshot']['parameters'])
 actual=ns['PROVIDERS'][key[1]](case)
 if actual['api']!=a['api']:issues.append('Formal function differs from tested API')
 case.id='__audit_unknown_key__'
 if ns['PROVIDERS'][key[1]](case)!='prior-preserved':issues.append('Prior route not preserved')
 row={'key':list(key),'accept_recommended':not issues,'decision':'recommend_accept' if not issues else 'needs_evidence','scope_reasons':s['classification']['reason'],'checker':ref(cpath),'verification':v,'inputs_preserved':c['inputs_preserved'],'seed':c['input_seed'],'checker_window':[c['started_at'],c['finished_at']],'guard':dict(ref(gpath),returncode=g['returncode'],metadata=ref(metadata),command=ref(command),batch=ref(batch),log=ref(guardlog)),'formal_factory':m.__name__+':make','formal_module':ref(mp),'provider':ref(ip),'frozen_adapter':ref(frozen),'cpu_stub_dispatch':actual,'source_provenance':{'attribution':ref(ap),'launches':launches,'log_hashes':record_sources,'kernel_hashes':a['kernel_hashes'],'checked_live_hashes':hashes,'original_compare':ref(P/'nn_recovery01/CURRENT_SOURCE_COMPARE.json')},'runtime_recipe':{'runner':ref(P/'nn_recovery01/run_case.py'),'OPS_NN_VENDOR':str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'),'ASCEND_CUSTOM_OPP_PATH':str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn'),'launcher':'baseline-run ops-nn; set vendor before Torch import'},'issues':issues}
 rows.append(row)
write('NN_REVIEW.json',{'rows':rows,'torch_npu_imported':False,'previous_provider_prefix_unchanged':prefix.rstrip()==(P/'nn_registration01/provider_init.before.py').read_text().rstrip()})
write('RECOMMENDED_ACCEPT.json',{'rows':rows,'partial':True})
write('HEARTBEAT.json',{'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'NN checks complete','rows':len(rows)})
print([(r['key'][1:],r['issues']) for r in rows])
