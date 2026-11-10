import sys,types,importlib,importlib.abc,pathlib,json
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/sgl_attention_register10';sys.path.insert(0,str(W/'TileOPs'))
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','triton','acl','sgl_kernel_npu'):raise RuntimeError('Real device imports forbidden: '+fullname)
sys.meta_path.insert(0,Block())
from benchmarks.ascend_all_pairs import registry,sgl
import benchmarks.ascend_baselines as base
# Explicit constructor-only substitutes, no tensor or numerical operations.
for name in ['torch','torch.nn','torch.nn.functional','benchmarks.api']:
 sys.modules[name]=types.ModuleType(name)
sys.modules['torch'].nn=sys.modules['torch.nn'];sys.modules['torch.nn'].functional=sys.modules['torch.nn.functional']
class Implementation:
 def __init__(self,run):self.run=run
sys.modules['benchmarks.api'].Implementation=Implementation
calls=[]
def fake_local_function(*args):
 calls.append({'args':list(args),'libraries':str(base.LIBRARIES)})
 return lambda *args:(_ for _ in ()).throw(AssertionError('Numerical execution forbidden'))
base.local_function=fake_local_function
# Poison generic adapter cache to prove the new binding never resolves it.
poison=types.ModuleType('dsa_adapter');poison.build=lambda _:(_ for _ in ()).throw(AssertionError('Generic module collision'));sys.modules['dsa_adapter']=poison
scope=json.loads((P/'SCOPE.json').read_text());results=[];saved=base.LIBRARIES
for row in scope:
 params=json.loads(row['source_reason'].split('本项参数=')[1])
 case=types.SimpleNamespace(id=row['case_id'],params=params,_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=row['operator'])))
 c=registry.candidate_factory(case,row['baseline'])()
 assert c.api=='sgl_kernel_npu.attention.decode_attention.decode_mla' and base.LIBRARIES==saved
 assert calls[-1]['libraries']==str(P/'libraries')
 assert c.source_sha256['reports/baseline-integration-700-20261026/sgl_attention_register10/dsa_adapter.py']=='bb4b09f8f840241f808c6a5df82bd3122038ec0e9d8e1e70a3247f870c152e59'
 results.append({'key':[row['baseline'],row['operator'],row['case_id']],'api':c.api,'source_sha256':c.source_sha256,'local_function':calls[-1]})
unknown=types.SimpleNamespace(id='__outside_exact_scope__',_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='DSADecodeWithKVCacheFwdOp')))
try:registry.candidate_factory(unknown,'sgl-kernel-npu')()
except base.UnsupportedCase:pass
else:raise AssertionError('Unregistered key unexpectedly constructed')
result={'rows':results,'generic_module_poison_ignored':True,'unknown_key_unsupported':True,'base_libraries_restored':True,'explicit_stubs':['torch empty modules','Implementation constructor','local_function recording stub'],'numerical_validation':False,'npu_imports':0}
(P/'CONSTRUCTOR_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
