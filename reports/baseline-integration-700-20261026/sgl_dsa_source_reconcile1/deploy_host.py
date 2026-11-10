import sys,pathlib,json,hashlib,fcntl,importlib,importlib.abc,types,ast,difflib
sys.dont_write_bytecode=True
payload=json.load(sys.stdin);W=pathlib.Path(payload['workspace']);P=W/'reports/baseline-integration-700-20261026/sgl_dsa_source_reconcile1';P.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,content in payload['files'].items():
 f=P/name
 if f.exists():assert f.read_text()==content,name
 else:f.write_text(content)
if payload['remote']:
 link=P/'libraries/sgl-kernel-npu';link.parent.mkdir(exist_ok=True)
 source=W/'reports/baseline-integration-700-20261026/sgl_attention_recovery32/libraries/sgl-kernel-npu'
 if not link.exists():link.symlink_to(source.resolve(),target_is_directory=True)
 for rel,h in payload['pins'].items():assert sha(link/rel)==h
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton','acl','sgl_kernel_npu'):raise RuntimeError('Forbidden real device import '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,str(W/'TileOPs'))
from benchmarks.ascend_all_pairs import registry,sgl
import benchmarks.ascend_baselines as base
module_name='benchmarks.ascend_all_pairs.sgl.integration700_sgl_dsa_source_reconcile1'
init=pathlib.Path(sgl.__file__);modpath=init.parent/'integration700_sgl_dsa_source_reconcile1.py'
def sig(f):return [f.__code__.co_filename,f.__code__.co_name,hashlib.sha256(f.__code__.co_code).hexdigest()]
with open('/tmp/baseline700-provider-sgl.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);importlib.reload(sgl)
 old=dict(sgl.PROVIDERS);before=init.read_bytes();assert b'# sgl_dsa_source_reconcile1:' not in before
 assert not modpath.exists()
 ast.parse(payload['module']);ast.parse(before.decode()+payload['append'])
 (P/'provider_registry.before.py').write_bytes(before)
 modpath.write_text(payload['module'])
 with init.open('ab') as f:f.write(payload['append'].encode())
 after=init.read_bytes();assert after==before+payload['append'].encode()
 (P/'provider_registry.after.py').write_bytes(after)
 (P/'REGISTRATION.patch').write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile=str(init),tofile=str(init))))
 importlib.reload(sgl)
 for op,f in old.items():
  if op!='DSADecodeWithKVCacheFwdOp':assert sig(sgl.PROVIDERS[op])==sig(f),op
 route=sgl.PROVIDERS['DSADecodeWithKVCacheFwdOp'];prev=dict(zip(route.__code__.co_freevars,[x.cell_contents for x in route.__closure__]))['previous']
 assert sig(prev)==sig(old['DSADecodeWithKVCacheFwdOp'])
 sentinel=object();assert sgl._dsa_sr1_wrap(lambda c:sentinel)(types.SimpleNamespace(id='__outside__')) is sentinel
 module=importlib.import_module(module_name)
 case=types.SimpleNamespace(id='mla-d512-h128-kv64k-bfloat16',params=payload['params'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='DSADecodeWithKVCacheFwdOp')))
 # Trace the actual disk registry to the new unique module before any dependencies.
 seen=[]
 class Reached(Exception):pass
 def trace(frame,event,arg):
  if event=='call' and frame.f_code.co_filename==str(modpath) and frame.f_code.co_name=='build':seen.append(str(modpath));raise Reached()
 sys.settrace(trace)
 try:registry.candidate_factory(case,'sgl-kernel-npu')()
 except Reached:pass
 finally:sys.settrace(None)
 assert seen==[str(modpath)]
 result={'cpu_only':True,'workspace':str(W),'actual_registry':str(pathlib.Path(registry.__file__)),'exact_key_route':seen,'outside_scope_fallback':True,'previous_dsa_provider_preserved':sig(prev),'other_providers_preserved':len(old)-1,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':sha(init),'module_sha256':sha(modpath),'append_only':True,'npu_runs':0,'device_import_blocker':True,'constructor_checked':False}
 if payload['remote']:
  for n in ['torch','torch.nn','torch.nn.functional','benchmarks.api']:sys.modules[n]=types.ModuleType(n)
  sys.modules['torch'].nn=sys.modules['torch.nn'];sys.modules['torch.nn'].functional=sys.modules['torch.nn.functional']
  class Implementation:
   def __init__(self,run):self.run=run
  sys.modules['benchmarks.api'].Implementation=Implementation
  calls=[]
  def stub(*args):
   calls.append({'args':list(args),'libraries':str(base.LIBRARIES)})
   return lambda *a:(_ for _ in ()).throw(AssertionError('Numerical execution forbidden'))
  base.local_function=stub
  poison=types.ModuleType('dsa_adapter');poison.build=lambda _:(_ for _ in ()).throw(AssertionError('Poison'));sys.modules['dsa_adapter']=poison
  saved=base.LIBRARIES;c=registry.candidate_factory(case,'sgl-kernel-npu')();assert base.LIBRARIES==saved
  assert c.api=='sgl_kernel_npu.attention.decode_attention.decode_mla' and c.cleanup is None and c.timing_mode=='host_roundtrip'
  assert c.implementation.run.__code__.co_filename==str(P/'dsa_adapter.py')
  # A fresh registry import must still execute the just-hashed adapter, ignoring generic poison.
  del sys.modules['benchmarks.ascend_all_pairs.sgl'];fresh=importlib.import_module('benchmarks.ascend_all_pairs.sgl');c2=registry.candidate_factory(case,'sgl-kernel-npu')();assert c2.api==c.api
  bad=types.SimpleNamespace(id=case.id,params=dict(case.params,dim_tail=64))
  try:module.build(bad)
  except base.UnsupportedCase:pass
  else:raise AssertionError('Bad params accepted')
  result.update(constructor_checked=True,constructor_api=c.api,source_hashes=c.source_sha256,original_api_resolver_calls=calls,generic_cache_poison_ignored=True,fresh_registry_import=True,semantic_parameter_guard=True,base_libraries_restored=True,library_pins=payload['pins'])
 (P/'FORMAL_REGISTRY_PROOF.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
