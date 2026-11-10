import sys,pathlib,json,hashlib,fcntl,types,importlib,datetime,os
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');D=W/'reports/baseline-integration-700-20261026/gdn_register1';A=D/'attempt_002_registration';A.mkdir(exist_ok=False)
assert json.loads((D/'attempt_001/READY.json').read_text())['ready']
cid='q35-397b-tp8-1024-7168-bfloat16';reg=W/'reports/baseline-integration-700-20261026/transformer_linear/REGISTERED.json';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,x):(A/n).write_text(json.dumps(x,indent=2)+'\n')
class Block:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','acl'):
   raise RuntimeError('DEVICE IMPORT BLOCKED '+fullname)
sys.meta_path.insert(0,Block());sys.path.insert(0,str(W/'TileOPs'))
registry=importlib.import_module('benchmarks.ascend_all_pairs.registry');provider=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer');factory=importlib.import_module('benchmarks.ascend_all_pairs.ops_transformer.integration700_transformer_linear_gdn')
# Device boundaries only: real registry, wrapper, factory loader and binding.build execute.
torch=types.ModuleType('torch');sys.modules['torch']=torch
api=types.ModuleType('benchmarks.api')
class Implementation:
 def __init__(self,fn):self.fn=fn
api.Implementation=Implementation;sys.modules['benchmarks.api']=api
ab=types.ModuleType('benchmarks.ascend_baselines')
class Candidate:
 def __init__(self,implementation,api,adaptation,hashes,**kw):self.implementation=implementation;self.api=api;self.hashes=hashes
ab.Candidate=Candidate;sys.modules['benchmarks.ascend_baselines']=ab
import ctypes
ctypes.CDLL=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('DSO loading forbidden'))
original_spec=importlib.util.spec_from_file_location;events=[]
def safe_spec(name,path,*a,**kw):
 spec=original_spec(name,path,*a,**kw)
 if name=='_integration700_transformer_linear_transport':
  real_exec=spec.loader.exec_module
  def exec_module(mod):
   real_exec(mod)
   mod.require_guard=lambda:events.append('guard stub: no device execution authorized')
   def init(self,op,torch,opapi,nnopbase):events.append({'transport_constructor_stub':op,'opapi':opapi,'nnopbase':nnopbase})
   mod.DirectACL.__init__=init
  spec.loader.exec_module=exec_module
 events.append({'load_module':name,'path':str(path),'sha256':sha(path)})
 return spec
importlib.util.spec_from_file_location=safe_spec
params=json.loads((factory.PRIVATE/'frozen_v1/CONTRACTS.json').read_text())[cid]['params'];case=types.SimpleNamespace(id=cid,params=params,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='GDNFwdOp')))
# CPU preflight actual registry with in-memory proposed allowlist; restores read_text immediately.
oldread=pathlib.Path.read_text
before_preview=reg.read_text()
def preview(self,*a,**kw):return json.dumps(json.loads(before_preview)+([cid] if cid not in json.loads(before_preview) else [])) if self==reg else oldread(self,*a,**kw)
pathlib.Path.read_text=preview
try: candidate=registry.candidate_factory(case,'ops-transformer')()
finally:pathlib.Path.read_text=oldread
assert candidate.api=='ops-transformer::aclnnChunkGatedDeltaRule'
save('PREFLIGHT.json',{'api':candidate.api,'events':events,'stub_boundaries':['torch module','Implementation/Candidate containers','require_guard','DirectACL.__init__'],'real_registry':registry.__file__,'real_factory':factory.__file__})
with open('/tmp/baseline700-provider-ops_transformer.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 raw=reg.read_bytes();old=json.loads(raw);assert isinstance(old,list) and all(isinstance(x,str) for x in old)
 (A/'REGISTERED.before.json').write_bytes(raw)
 new=old+([cid] if cid not in old else [])
 if new!=old:
  # Latest bytes were read under the shared provider lock; preserve all old entries and order.
  with reg.open('w') as f:json.dump(new,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 actual=json.loads(reg.read_text());assert actual==new and [x for x in actual if x!=cid]==[x for x in old if x!=cid]
 (A/'REGISTERED.after.json').write_bytes(reg.read_bytes())
 events.clear();candidate=registry.candidate_factory(case,'ops-transformer')();assert candidate.api=='ops-transformer::aclnnChunkGatedDeltaRule'
 save('REGISTRATION.json',{'key':['ops-transformer','GDNFwdOp',cid],'registered':True,'added':cid not in old,'before_count':len(old),'after_count':len(actual),'preserved_other_keys':True,'before_sha256':hashlib.sha256(raw).hexdigest(),'after_sha256':sha(reg),'path':str(reg),'lock':'/tmp/baseline700-provider-ops_transformer.lock','actual_registry':registry.__file__,'registry_sha256':sha(registry.__file__),'provider':provider.__file__,'provider_sha256':sha(provider.__file__),'factory':factory.__file__,'factory_sha256':sha(factory.__file__),'api':candidate.api,'binding_hashes':{p:h for p,h in candidate.hashes.items() if '/frozen_v1/chunk_package/' in p},'events':events,'cpu_only':True,'numerical_rerun':False,'device_imports':[],'stub_boundaries':['torch module','Implementation/Candidate containers','require_guard','DirectACL.__init__'],'guard_returncode_preserved':1,'checker_status_preserved':'source_changed'})
print(json.dumps({'registered':True,'added':cid not in old,'counts':[len(old),len(actual)],'real_binding':candidate.implementation.fn.__code__.co_filename,'events':events}))
