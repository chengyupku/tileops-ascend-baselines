import pathlib,json,hashlib,datetime,fcntl,sys,types,os,shutil,sqlite3
sys.dont_write_bytecode=True
W=pathlib.Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'fft_window_first1';N=W/'TileOPs/benchmarks/ascend_all_pairs/ops_math';read=lambda p:json.loads(p.read_text());cache={}
def sha(p):
 p=pathlib.Path(p)
 if str(p) not in cache:cache[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return cache[str(p)]
def put(p,v):
 t=p.with_name(p.name+'.parent.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
for x in read(P/'EVIDENCE_INDEX.json'):assert sha(P/x['path'])==x['sha256'],x['path']
c=read(P/'checker.json');assert c['status']=='verified' and c['verification']['checked_outputs']==c['verification']['total_outputs']==1 and c['inputs_preserved'] and not c['provider_source_changed'] and not c['oracle_source_changed']
assert c['input_seed']==int.from_bytes(hashlib.sha256(b'FFTC2CFwdOp::latency-4k-complex64').digest()[:4],'big')
assert read(P/'GUARD_RESULT.json')['returncode']==read(P/'CHILD_RESULT.json')['returncode']==0
assert read(P/'GUARD_METADATA.json')['owner']=='baseline700-20261026/fft_window_first1'
for field in ['source_sha256','oracle_source_sha256']:
 for p,h in c[field].items():assert sha(W/p)==h,p
assert sha(W/'TileOPs/scripts/ascend_pair_check.py')==sha(P/'ascend_pair_check.snapshot.py')
for r in read(P/'SOURCE_COMPARE.json'):assert r['matches'] and r['old']==r['current']==sha(W/r['path'])
native=[]
for x in read(P/'NATIVE_EVIDENCE_INDEX.json'):
 assert sha(x['path'])==x['sha256'];ls=pathlib.Path(x['path']).read_text(errors='replace').splitlines()
 for a,b in x['ranges']:native.extend(ls[a-1:b])
joined='\n'.join(native);assert 'RunCpuKernel[STFT] success' in joined and 'windowOptional: aclTensor' in joined and 'opType[STFT]' in joined and 'libcpu_kernels.so' in joined
payload=read(P/'PAYLOADS.json');assert len(payload)==1 and payload[0]['full_output_elements']==4096 and payload[0]['input_preserved']
assert read(P/'WINDOW_BINDING.json')['all_one_plus_zero_j']
# Runtime pin complete tested source/runtime dependencies; frozen adapter remains byte-identical.
put(P/'REGISTER_DEPENDENCIES.json',c['source_sha256']);dep=sha(P/'REGISTER_DEPENDENCIES.json');vendor=W/'.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math'
module=f'''"""Exact original aclStft route; native system AiCPU dependency disclosed."""
from pathlib import Path
import hashlib, json, os, types
_KEY = ('FFTC2CFwdOp', 'latency-4k-complex64')
_P = Path({str(P)!r})
_VENDOR = {str(vendor)!r}
def _load():
    data=(_P/'adapter.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!={sha(P/'adapter.py')!r}: raise RuntimeError('Frozen adapter changed')
    m=types.ModuleType('fft_window_first1_frozen');m.__file__=str(_P/'adapter.py');exec(compile(data,m.__file__,'exec'),m.__dict__);return m
def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name,case.id)!=_KEY: raise UnsupportedCase('No exact receipt')
    if os.environ.get('ASCEND_CLASSIC_MATH_VENDOR')!=_VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=_VENDOR: raise UnsupportedCase('Select recorded classicpilot_math before runtime initialization')
    data=(_P/'REGISTER_DEPENDENCIES.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!={dep!r}: raise RuntimeError('Dependency manifest changed')
    for relative,h in json.loads(data).items():
        if hashlib.sha256((Path({str(W)!r})/relative).read_bytes()).hexdigest()!=h: raise RuntimeError('Dependency changed: '+relative)
    return _load().make(case)
'''
append='''
# Exact full original API receipt: baseline700/fft_window_first1 (system AiCPU STFT).
from . import integration700_fft_window_first1 as _fft_window_first1
_fft_window_first1_prior = PROVIDERS.get('FFTC2CFwdOp')
def _fft_window_first1_route(case, _prior=_fft_window_first1_prior):
    if (case._call.signature.name, case.id) == _fft_window_first1._KEY:
        return _fft_window_first1.make(case)
    if _prior is not None:
        return _prior(case)
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase('No exact FFT route')
PROVIDERS['FFTC2CFwdOp'] = _fft_window_first1_route
'''
sys.path.insert(0,str(W/'TileOPs'))
with open('/tmp/baseline700-provider-ops_math.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);old=(N/'__init__.py').read_text();assert '_fft_window_first1_route' not in old;compile(module,'newmodule','exec');compile(old+append,'registry','exec');(P/'PARENT_PROVIDER_BEFORE.py').write_text(old);(N/'integration700_fft_window_first1.py').write_text(module);(P/'integration700_fft_window_first1.py').write_text(module);(N/'__init__.py').write_text(old+append)
 try:
  from benchmarks.ascend_all_pairs import registry,ops_math
  m=ops_math._fft_window_first1;assert sha(pathlib.Path(m._load().__file__))==sha(P/'adapter.py');m._load=lambda:types.SimpleNamespace(make=lambda c:('expected',c.id));os.environ['ASCEND_CLASSIC_MATH_VENDOR']=str(vendor);os.environ['ASCEND_CUSTOM_OPP_PATH']=str(vendor)
  case=types.SimpleNamespace(id='latency-4k-complex64',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='FFTC2CFwdOp')));assert registry.candidate_factory(case,'ops-math')()==('expected',case.id)
  case.id='other';assert ops_math._fft_window_first1_route(case,_prior=lambda c:'prior')=='prior';assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
 except BaseException:
  (N/'__init__.py').write_text(old);raise
now=datetime.datetime.now(datetime.timezone.utc).isoformat();key=['ops-math','FFTC2CFwdOp','latency-4k-complex64'];qual='Original ops-math complete aclStft host API selects system CANN AiCPU STFT, proven by original source and raw host/device registration/load/launch/success. This is an original dependency, not adapter mathematical fallback or custom vendor kernel. Device libcpu_kernels.so bytes inaccessible on host; no independent device-binary SHA256 claim. Correctness-only, not AI Core benchmark.'
review={'at':now,'status':'accepted','key':key,'accepted_added':1,'hashes_checked':len(cache),'native_indexed_lines_checked':len(native),'original_reference_checked_outputs':1,'full_output_elements':4096,'input_preserved':True,'guard_rc':0,'child_rc':0,'guard_wall_s':read(P/'GUARD_RESULT.json')['elapsed_s'],'formal_registry_exact_key_verified':True,'registry_sha256':sha(N/'__init__.py'),'module_sha256':sha(N/'integration700_fft_window_first1.py'),'original_boundary':qual,'qualification':qual,'new_device_runs_by_parent':0,'successful_case_reruns':0}
with open(str(D/'ACCEPTED.json')+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);a=read(D/'ACCEPTED.json');assert len(a['rows'])==321 and key not in [x['key'] for x in a['rows']];shutil.copy2(D/'ACCEPTED.json',D/('ACCEPTED.before_fft1_'+now.replace(':','-')+'.json'));a['rows'].append({'key':key,'accepted':True,'at':now,'host':'local','checker':str(P/'checker.json'),'guard':str(P/'GUARD_RESULT.json'),'parent_review':str(P/'PARENT_REVIEW.json'),'qualification':qual});a['updated_at']=now;put(D/'ACCEPTED.json',a)
review['accepted_total']=322;put(P/'PARENT_REVIEW.json',review);shutil.copy2('/tmp/accept_fft1.py',D/'parent_accept_fft1.py')
p=D/'PLAN.json';x=read(p)
for w in x['workers']:
 if w['name']=='fft_window_first1':w.update(active_assignment=False,completed_parent_accepted=1)
x['updated_at']=now;put(p,x)
p=D/'STATE.json';s=read(p);s.update(completed=322,parent_accepted_completed=322,last_status_accepted=322,last_status_at=now,active_workers=0,last_acceptance_review=str(P/'PARENT_REVIEW.json'));s['closed_success_scopes'].append('fft_window_first1');s['ownership_audit']={'unique':700,'duplicates':0,'accepted':322,'root':373,'diagnostic_reserved':0,'pending_registration':0,'active_execution':0,'pending_native':5};put(p,s)
M=W/'.agent_memory/THREADS/baseline-inspector-20261023/STATE.json'
with open(str(M)+'.lock','a') as l,open(M.parent/'STATE.lock','a') as b:
 fcntl.flock(l,fcntl.LOCK_EX);fcntl.flock(b,fcntl.LOCK_EX);shutil.copy2(M,M.with_name('STATE.before_fftaccept_'+now.replace(':','-')+'.json'));m=read(M);m['integration_700_20261026'].update({k:s[k] for k in ['parent_accepted_completed','last_status_at','active_workers','last_acceptance_review','closed_success_scopes','ownership_audit']});put(M,m)
print({'accepted':322,'remaining':378,'hashes_checked':len(cache),'native_lines':len(native)})
