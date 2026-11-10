import sys,json,pathlib,hashlib,fcntl,importlib,importlib.abc,types,difflib
sys.dont_write_bytecode=True
payload=json.load(sys.stdin)
W=pathlib.Path('/opt/tileops-all-pairs/workspace'); P=W/'reports/baseline-integration-700-20261026/sgl_attention_register10';P.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Read-only dependency audit before mutation.
checks=[]
for f,v in payload['remote_hashes'].items():
 p=pathlib.Path(f);assert sha(p)==v,f;checks.append({'path':f,'sha256':v})
for name,data in payload['files'].items():
 target=P/name
 if target.exists():assert target.read_text()==data,name
 else:target.write_text(data)
libs=P/'libraries';libs.mkdir(exist_ok=True)
original=W/'reports/baseline-integration-700-20261026/sgl_attention_recovery32/libraries/sgl-kernel-npu'
link=libs/'sgl-kernel-npu'
if not link.exists():link.symlink_to(original.resolve(),target_is_directory=True)
pins=[]
for x in payload['source_hashes']:
 f=pathlib.Path(x['path']);rel=str(f).split('/libraries/sgl-kernel-npu/')
 if len(rel)==2:
  pinned=link/rel[1];assert sha(pinned)==x['sha256'];pins.append({'path':str(pinned),'resolved':str(pinned.resolve()),'sha256':x['sha256']})
(P/'PINNED_LIBRARY.json').write_text(json.dumps(pins,indent=2))
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl','triton','sgl_kernel_npu'):
   raise RuntimeError('FORBIDDEN_DEVICE_IMPORT:'+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,str(W/'TileOPs'))
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs import sgl
import benchmarks.ascend_baselines as base
class Reached(Exception):pass
def route(op,cid):
 case=types.SimpleNamespace(id=cid,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
 found=[]
 def trace(frame,event,arg):
  if event=='call' and frame.f_code.co_name in ('build','write','varlen','mla_decode','mha_paged','prefill_append'):
   f=frame.f_code.co_filename
   if 'integration700_sgl_attention_register10.py' not in f:
    found.append([f,frame.f_code.co_name]);raise Reached()
  return trace
 sys.settrace(trace)
 try:registry.candidate_factory(case,'sgl-kernel-npu')()
 except Reached:pass
 finally:sys.settrace(None)
 assert found,(op,cid)
 return found[0]
# Serialize read-latest, append and verify under existing host coordination lock.
with open('/tmp/baseline700-provider-sgl.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 importlib.reload(sgl)
 old=dict(sgl.PROVIDERS);prior=[]
 for name,value in list(vars(sgl).items()):
  if name.endswith('_keys') and isinstance(value,dict):
   for op,keys in value.items():
    for cid in keys:prior.append((op,cid))
 prior=list(dict.fromkeys(prior));before={op+'::'+cid:route(op,cid) for op,cid in prior}
 init=pathlib.Path(sgl.__file__);original_bytes=init.read_bytes();marker=b'# sgl_attention_register10:';assert marker not in original_bytes
 module=init.parent/'integration700_sgl_attention_register10.py'
 assert not module.exists()
 module.write_text(payload['module'])
 append=payload['append'].encode();(P/'provider_init.before.py').write_bytes(original_bytes)
 with init.open('ab') as f:f.write(append)
 (P/'REGISTRATION.patch').write_text(''.join(difflib.unified_diff(original_bytes.decode().splitlines(True),(original_bytes+append).decode().splitlines(True),fromfile=str(init),tofile=str(init))))
 importlib.reload(sgl)
 assert init.read_bytes()==original_bytes+append
 exact=[]
 for cid in payload['keys']:
  entry=route('DSADecodeWithKVCacheFwdOp',cid);assert entry==[str(P/'dsa_adapter.py'),'build'];exact.append({'case_id':cid,'entry':entry})
 after={op+'::'+cid:route(op,cid) for op,cid in prior};assert before==after
 # Non-target provider identities across reload naturally change; compare defining code instead.
 for op in old:
  if op!='DSADecodeWithKVCacheFwdOp':assert sgl.PROVIDERS[op].__code__ is old[op].__code__ or (sgl.PROVIDERS[op].__code__.co_code==old[op].__code__.co_code and sgl.PROVIDERS[op].__code__.co_filename==old[op].__code__.co_filename)
 sentinel=object();assert sgl._ar10_wrap(lambda c:sentinel)(types.SimpleNamespace(id='__outside__')) is sentinel
 # Fresh package import, including previously cached unique binding, must route identically.
 del sys.modules['benchmarks.ascend_all_pairs.sgl']; fresh=importlib.import_module('benchmarks.ascend_all_pairs.sgl')
 for cid in payload['keys']:assert route('DSADecodeWithKVCacheFwdOp',cid)==[str(P/'dsa_adapter.py'),'build']
 result={'cpu_only':True,'actual_registry':registry.__file__,'device_imports_blocked':['torch','torch_npu','acl','triton','sgl_kernel_npu'],'exact_routes':exact,'prior_exact_routes_unchanged':before,'prior_exact_count':len(before),'fallback_sentinel':True,'fresh_import':True,'numerical_reexecution':False,'constructor_validation':'Trace stops at frozen build entry before runtime imports; not numerical proof.','remote_hashes_verified':len(checks),'source_dependencies_pinned':pins,'shared_init_before_sha256':hashlib.sha256(original_bytes).hexdigest(),'shared_init_after_sha256':sha(init),'unique_module_sha256':sha(module),'frozen_adapter_sha256':sha(P/'dsa_adapter.py'),'append_only':True,'library_deployment':'Independent private libraries namespace symlinks to hash-pinned pristine original tree, shared with prior accepted SGL imports to preserve canonical origin; every pinned dependency checked before build.'}
 (P/'DEVICE_FREE_CHECKS.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
