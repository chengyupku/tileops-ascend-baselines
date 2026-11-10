import pathlib,json,hashlib,datetime,fcntl,sys,types,os,shutil
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_prod_shortpath1';N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';read=lambda p:json.loads(p.read_text());cache={}
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  cache[str(p)]=h.hexdigest()
 return cache[str(p)]
def put(p,x):
 q=p.with_name(p.name+'.parent.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');os.replace(q,p)
for r in read(P/'DELIVERY.json')['files']:assert sha(P/r['path'])==r['sha256'],r['path']
c=read(P/'checker.json');assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed'];assert c['input_seed']==int.from_bytes(hashlib.sha256(b'ProdFwdOp::long-seq-bfloat16').digest()[:4],'big');assert c['case_snapshot']['declared_tensors']['output']==[[64],'bfloat16']
for field in ['source_sha256','oracle_source_sha256']:
 for p,h in c[field].items():assert sha(W/p)==h,p
for p,r in read(P/'SOURCE_COMPARE.json')['files'].items():assert r['expected']==r['actual']==sha(W/p),p
assert read(P/'GUARD_RESULT.json')['returncode']==read(P/'CHILD_RESULT.json')['returncode']==0
gm=read(P/'GUARD_METADATA.json');assert gm['owner']=='baseline700-20261026/nn_prod_shortpath1' and gm['acquired_at']<c['started_at'];pid=gm['child']['pid'];events=read(P/'NATIVE_PROD_EVIDENCE.json');refs={}
for e in events:refs.setdefault(e['file'],{})[e['line']]=e['text']
for p,lines in refs.items():
 raw=(W/p).read_text(errors='replace').splitlines()
 for n,t in lines.items():assert raw[n-1]==t
man=read(P/'CACHE_ELF_MANIFEST.json');kernel=man['kernel'];raw='\n'.join(e['text'] for e in events);assert kernel in raw and 'StreamLaunchKernelV1' in raw and f'({pid},python)' in raw and 'allpairs_nn' in raw
for a in man['artifacts']:assert sha(a['source'])==sha(a['saved'])==a['sha256']
elf=next(a for a in man['artifacts'] if a['saved'].endswith('.o'));meta=next(a for a in man['artifacts'] if a['saved'].endswith('.json'));assert read(pathlib.Path(meta['saved']))['sha256']==elf['sha256']
assert sha(P/'runner.py')==read(P/'PARENT_PATH_PREFLIGHT.json')['parent_prepared_runner_sha256'];assert sha(P/'frozen/TileOPs/scripts/ascend_pair_check.py')==sha(W/'TileOPs/scripts/ascend_pair_check.py')
put(P/'REGISTER_DEPENDENCIES.json',c['source_sha256']);template=(D/'fft_window_first1/integration700_fft_window_first1.py').read_text();old_vendor=str(W/'.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math');vendor=str(W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn')
module=template.replace(str(D/'fft_window_first1'),str(P)).replace("('FFTC2CFwdOp', 'latency-4k-complex64')","('ProdFwdOp', 'long-seq-bfloat16')").replace('fft_window_first1_frozen','nn_prod_shortpath1_frozen').replace(old_vendor,vendor).replace('ASCEND_CLASSIC_MATH_VENDOR','OPS_NN_VENDOR').replace(sha(D/'fft_window_first1/adapter.py'),sha(P/'adapter.py')).replace(sha(D/'fft_window_first1/REGISTER_DEPENDENCIES.json'),sha(P/'REGISTER_DEPENDENCIES.json')).replace('Exact original aclStft route; native system AiCPU dependency disclosed.','Exact original UnsortedSegmentProd; verified full output and original native cache ELF.')
append='''
# Exact original UnsortedSegmentProd receipt: baseline700/nn_prod_shortpath1.
from . import integration700_nn_prod_shortpath1 as _nn_prod_shortpath1
_nn_prod_shortpath1_prior = PROVIDERS.get('ProdFwdOp')
def _nn_prod_shortpath1_route(case, _prior=_nn_prod_shortpath1_prior):
    if (case._call.signature.name, case.id) == _nn_prod_shortpath1._KEY:
        return _nn_prod_shortpath1.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact Prod route')
PROVIDERS['ProdFwdOp'] = _nn_prod_shortpath1_route
'''
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_nn.lock','a') as l:
 fcntl.flock(l,fcntl.LOCK_EX);old=(N/'__init__.py').read_text();assert '_nn_prod_shortpath1_route' not in old;compile(old+append,'registry','exec');compile(module,'module','exec');(P/'PARENT_PROVIDER_BEFORE.py').write_text(old);(N/'integration700_nn_prod_shortpath1.py').write_text(module);(P/'integration700_nn_prod_shortpath1.py').write_text(module);(N/'__init__.py').write_text(old+append)
 try:
  from benchmarks.ascend_all_pairs import registry,ops_nn
  m=ops_nn._nn_prod_shortpath1;assert sha(pathlib.Path(m._load().__file__))==sha(P/'adapter.py');m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id));os.environ['OPS_NN_VENDOR']=vendor;os.environ['ASCEND_CUSTOM_OPP_PATH']=vendor
  case=types.SimpleNamespace(id='long-seq-bfloat16',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='ProdFwdOp')));assert registry.candidate_factory(case,'ops-nn')()==('expected',case.id);case.id='unknown';assert ops_nn._nn_prod_shortpath1_route(case,_prior=lambda c:'prior')=='prior';assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
now=datetime.datetime.now(datetime.timezone.utc).isoformat();key=['ops-nn','ProdFwdOp','long-seq-bfloat16'];review={'at':now,'status':'accepted','key':key,'hashes_checked':len(cache),'raw_native_lines_verified':sum(map(len,refs.values())),'native_kernel':kernel,'cached_ELF_sha256':elf['sha256'],'native_compile':'Graph build reused original matching compiled ELF cache; no fresh kernel compilation claimed.','boundary':'Transpose contiguous, zero segment metadata, num_segments=1; one original complete UnsortedSegmentProd; output reshape only. Full64 output unchanged canonical reference/seed/tolerance,input preserved.','guard_rc':0,'child_rc':0,'guard_wall_s':read(P/'GUARD_RESULT.json')['elapsed_s'],'formal_registry_exact_key_verified':True,'module_sha256':sha(N/'integration700_nn_prod_shortpath1.py'),'registry_sha256':sha(N/'__init__.py'),'numeric_reruns_by_parent':0,'accepted_added':1}
with open(str(D/'ACCEPTED.json')+'.lock','a') as l:
 fcntl.flock(l,fcntl.LOCK_EX);a=read(D/'ACCEPTED.json');assert len(a['rows'])==331 and key not in [r['key'] for r in a['rows']];shutil.copy2(D/'ACCEPTED.json',D/('ACCEPTED.before_prod_'+now.replace(':','-')+'.json'));a['rows'].append({'key':key,'accepted':True,'at':now,'host':'local','checker':str(P/'checker.json'),'guard':str(P/'GUARD_RESULT.json'),'parent_review':str(P/'PARENT_REVIEW.json')});a['updated_at']=now;put(D/'ACCEPTED.json',a)
review['accepted_total']=332;put(P/'PARENT_REVIEW.json',review);shutil.copy2('/tmp/accept_prod_short.py',D/'parent_accept_prod_short.py')
p=D/'PLAN.json';x=read(p)
for w in x['workers']:
 if w['name']=='nn_prod_shortpath1':w.update(active_assignment=False,completed_parent_accepted=1)
put(p,x);s=read(D/'STATE.json');s.update(completed=332,parent_accepted_completed=332,last_status_accepted=332,last_status_at=now,active_workers=0,last_acceptance_review=str(P/'PARENT_REVIEW.json'));s['ownership_audit'].update(accepted=332,active_execution=0);s['closed_success_scopes'].append('nn_prod_shortpath1');put(D/'STATE.json',s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as l,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(l,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_prodaccept_'+now.replace(':','-')+'.json'));m=read(M);m['integration_700_20261026'].update({k:s[k] for k in ['parent_accepted_completed','last_status_at','active_workers','last_acceptance_review','ownership_audit','closed_success_scopes']});put(M,m)
print({'accepted':332,'remaining':368,'hashes_checked':len(cache),'native_lines_checked':review['raw_native_lines_verified']})
