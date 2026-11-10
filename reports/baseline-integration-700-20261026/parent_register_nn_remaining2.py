from pathlib import Path
import json,hashlib,datetime,fcntl,os,shutil,re,subprocess
W=Path('/root/dyq/workspace');D=W/'reports/baseline-integration-700-20261026';P=D/'nn_remote_remaining6';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,j):
 p=Path(p);t=p.with_suffix('.tmp-parent');t.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');os.replace(t,p)
rows=[];hashes={};events_checked=0;source_equal=[]
for name,sub in [('case0_inplace','inplace_add_rms_norm'),('case4','batch_norm_v3')]:
 A=P/name;c=read(A/'checker.json');g=read(A/'GUARD_RESULT.json');v=c['verification'];assert c['status']=='verified' and v['checked_outputs']==v['total_outputs']>0 and c['inputs_preserved'] and not c['oracle_source_changed'];assert not any(c.get(x) for x in ('error','sync_error','cleanup_error'));assert g['returncode']==0;key=[c['library'],c['op'],c['case_id']];assert c['input_seed']==int.from_bytes(hashlib.sha256((key[1]+'::'+key[2]).encode()).digest()[:4],'big');assert c['verifier_sha256']==sha(W/'TileOPs/scripts/ascend_pair_check.py')
 for p,h in read(A/'FILES_SHA256.json').items():hashes[str(P/p)]=h
 for field in ['source_sha256','oracle_source_sha256']:
  for p,h in c[field].items():hashes[str(Path(p) if Path(p).is_absolute() else W/p)]=h
 vendor=W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn/op_impl/ai_core/tbe/allpairs_nn_impl/ascendc'/sub;original=W/'ascend_baselines/ops-nn/norm'/sub/'op_kernel'
 for p in vendor.rglob('*'):
  if p.is_file() and p.suffix in ('.cpp','.h'):
   op=original/p.relative_to(vendor)
   if not op.exists():
    matches=list(original.rglob(p.name));assert len(matches)==1,(p,matches);op=matches[0]
   assert sha(p)==sha(op),(p,op);source_equal.append(str(p))
 raw={json.loads(l)['text'] for l in (A/'native.jsonl').read_text().splitlines() if l.strip()};ev=read(A/'NATIVE_TARGET_EVENTS.json');launches=[];rawcache={}
 for e in ev:
  if e['text'] not in raw:
   if e['file'] not in rawcache:rawcache[e['file']]=Path(e['file']).read_text(errors='replace').splitlines()
   assert rawcache[e['file']][e['line']-1].rstrip()==e['text'].rstrip()
  events_checked+=1
  if 'StreamLaunchKernelV1:' in e['text']:
   m=re.search(r'(2026-\d\d-\d\d)-(\d\d:\d\d:\d\d)\.(\d{3})\.(\d{3})',e['text']);assert m
   t=m[1]+'T'+m[2]+'.'+m[3]+m[4]+'+00:00';assert c['started_at']<=t<=c['finished_at'];launches.append(e['text'])
 assert launches
 gm=read(A/'GUARD_METADATA.json');rows.append({'key':key,'checker':str(A/'checker.json'),'guard':str(A/'GUARD_RESULT.json'),'full_outputs':v,'guard_returncode':0,'native_launches':launches,'lock_wait_seconds':(datetime.datetime.fromisoformat(gm['acquired_at'])-datetime.datetime.fromisoformat(gm['requested_at'])).total_seconds()})
for p,h in hashes.items():assert sha(p)==h,p
# Registry changes are parent-authorized integration; perform CPU preflight with runtime constructors stubbed.
module_name='integration700_nn_remote_remaining6';provider=W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn';target=provider/(module_name+'.py');append=(P/'REGISTRY_APPEND.proposed.py').read_text();append='\n'+append[append.index('from . import'):]
probe='''import sys,pathlib,json,types,importlib.abc
sys.dont_write_bytecode=True
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','triton','acl'):raise RuntimeError('Device import blocked '+name)
sys.meta_path.insert(0,Block());W=pathlib.Path('/root/dyq/workspace');P=W/'reports/baseline-integration-700-20261026/nn_remote_remaining6';sys.path[:0]=[str(W/'TileOPs'),str(W/'TileOPs/src'),str(W)]
from benchmarks.ascend_all_pairs import registry,ops_nn
from benchmarks.ascend_all_pairs.ops_nn import graph
class StubGraph:
 def __init__(self,lib):self.vendor=W/'.cache/all-pairs/ops_nn/install3/vendors/allpairs_nn';assert lib=='ops-nn'
 def cleanup(self):pass
graph.GraphCall=StubGraph
sys.modules['torch']=types.ModuleType('torch')
api=types.ModuleType('benchmarks.api');api.Implementation=lambda fn:types.SimpleNamespace(run=fn);sys.modules['benchmarks.api']=api
before=dict(ops_nn.PROVIDERS)
if APPEND is not None:exec(compile(APPEND,str(W/'TileOPs/benchmarks/ascend_all_pairs/ops_nn/__init__.py'),'exec'),ops_nn.__dict__)
records=[]
for name,x in json.loads((P/'REGISTRATION_BINDINGS.json').read_text()).items():
 c=json.loads(pathlib.Path(x['checker']).read_text());case=types.SimpleNamespace(id=c['case_id'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=c['op'])),params=c['case_snapshot']['parameters'])
 candidate=registry.candidate_factory(case,'ops-nn')();assert candidate.api==c['api'];assert pathlib.Path(candidate.implementation.run.__code__.co_filename)==P/name/'adapter.py'
 records.append({'key':x['key'],'api':candidate.api,'adapter':candidate.implementation.run.__code__.co_filename})
for op in ('FusedAddRMSNormFwdOp','BatchNormFwdOp'):
 route=ops_nn.PROVIDERS[op];prior=route.__defaults__[0]
 assert prior is not None
 sentinel=object();old=route.__defaults__;route.__defaults__=(lambda case:sentinel,)
 try:assert route(types.SimpleNamespace(id='__outside_scope__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))) is sentinel
 finally:route.__defaults__=old
for op,fn in before.items():
 if op not in ('FusedAddRMSNormFwdOp','BatchNormFwdOp'):assert ops_nn.PROVIDERS[op] is fn
assert not any(s in pathlib.Path('/proc/self/maps').read_text() for s in ('libascendcl.so','libtorch_npu.so'))
print(json.dumps({'rows':records,'fallbacks_preserved':2,'device_libraries_loaded':False,'runtime_constructors_stubbed':True,'candidate_execution':False}))
'''
with open('/tmp/baseline700-provider-ops_nn.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);init=provider/'__init__.py';before=init.read_bytes();assert module_name.encode() not in before and not target.exists();target.write_bytes((P/(module_name+'.py')).read_bytes())
 q=subprocess.run(['python3','-S','-'],input='APPEND='+repr(append)+'\n'+probe,text=True,capture_output=True);assert q.returncode==0,q.stderr
 (P/'PARENT_provider_init.before.py').write_bytes(before);init.write_bytes(before+append.encode());(P/'PARENT_provider_init.after.py').write_bytes(init.read_bytes())
 q=subprocess.run(['python3','-S','-'],input='APPEND=None\n'+probe,text=True,capture_output=True);assert q.returncode==0,q.stderr;cpu=json.loads(q.stdout)
reg={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'module':str(target),'module_sha256':sha(target),'provider':str(init),'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(init),'append_only':True,'keys':[r['key'] for r in rows],'cpu_check':cpu,'npu_runs':0};put(P/'PARENT_REGISTRATION.json',reg)
now=reg['at'];review={'at':now,'qualified_rows':rows,'artifact_and_source_hashes_verified':len(hashes),'installed_original_kernel_sources_equal':len(source_equal),'native_events_verified':events_checked,'original_boundary':'Whole original InplaceAddRmsNorm using cloned input/output aliases and BatchNormV3 with complete output/state. Original source vendor selected in compile log, unchanged kernel source equality and named native launches within checker windows. No mathematical fallback/compiler/kernel/tolerance changes.','registration':str(P/'PARENT_REGISTRATION.json'),'new_npu_runs':0,'dispatch_correction':'Parent permitted local device execution without widening worker local shared-provider write scope; parent completed authorized registration directly. Future host amendments must include corresponding provider write scope.'};rp=P/'PARENT_REVIEW.json'
with open(str(D/'ACCEPTED.json')+'.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ledger=read(D/'ACCEPTED.json');by={tuple(r['key']):r for r in ledger['rows']};old=len(by);shutil.copy2(D/'ACCEPTED.json',D/('ACCEPTED.before_nn_remaining2_'+now.replace(':','-')+'.json'))
 for r in rows:by.setdefault(tuple(r['key']),dict(r,at=now,accepted=True,host='local',parent_review=str(rp),original_checker_status='verified'))
 ledger.update(rows=list(by.values()),updated_at=now);put(D/'ACCEPTED.json',ledger)
review.update(accepted_added=len(by)-old,accepted_total=len(by));put(rp,review)
print(json.dumps({'accepted_added':len(by)-old,'accepted_total':len(by),'hashes':len(hashes),'original_sources':len(source_equal),'native_events':events_checked}))
