"""Device-free integration checks. Never import real torch/torch_npu/TileLang/TVM."""
import ast, builtins, hashlib, importlib, importlib.abc, json, pathlib, sys, types, os
W=pathlib.Path('/opt/tileops-all-pairs/workspace')
T=W/'reports/baseline-integration-20261026/register_server1'
D=W/'TileOPs/benchmarks/ascend_all_pairs/tla'
blocked=[]
class BlockDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','tilelang','tvm','triton'):
   blocked.append(fullname)
   raise AssertionError('Real device/compiler import forbidden: '+fullname)
sys.meta_path.insert(0,BlockDevice())
for name,path in [('benchmarks',W/'TileOPs/benchmarks'),('benchmarks.ascend_all_pairs',D.parent)]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
registry=importlib.import_module('benchmarks.ascend_all_pairs.registry')
tla=importlib.import_module('benchmarks.ascend_all_pairs.tla')
v=importlib.import_module('benchmarks.ascend_all_pairs.tla.verified_where')
v.inspect_pins()
for p in D.glob('*.py'):compile(p.read_text(),str(p),'exec')
# Compare the original module structurally after removing only the new dispatch.
old=ast.parse((T/'backups/__init__.py').read_text());new=ast.parse((D/'__init__.py').read_text())
fn=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_make')
fn.body=fn.body[2:]
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
assert set(tla.PROVIDERS)==set(tla.IMPLEMENTED)
# All historical op routes still reach the old prepare + factory for unknown IDs.
seen=[]
native=types.ModuleType(tla.__name__+'.native');native.prepare_deployment=lambda:seen.append('prepare')
adapters=types.ModuleType(tla.__name__+'.adapters');adapters.factory=lambda op,case:('historical',op,case.id)
sys.modules[native.__name__]=native;sys.modules[adapters.__name__]=adapters
case=lambda op,cid,p=None:types.SimpleNamespace(id=cid,params=p or {},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)))
for op in tla.PROVIDERS:
 c=case(op,'unknown-case');assert registry.candidate_factory(c,'tilelang-ascend')()==('historical',op,'unknown-case')
assert len(seen)==len(tla.PROVIDERS)
# Verified ID resolves through public registry, while the factory preserves exact args.
original_prepare=v._prepare
mock_native=object();mock_tl=object();mock_out=object()
class Adapter:
 def factory(self,*args):return args
v._prepare=lambda:(mock_native,mock_tl,Adapter(),mock_out)
c=case('WhereFwdOp','256M-float16',v.config()['params'])
assert registry.candidate_factory(c,'tilelang-ascend')()==(c,'WhereFwdOp',mock_native,mock_tl,mock_out)
try:v.factory(case('WhereFwdOp','256M-float16',{'output':[[1,1],'float16']}))
except ValueError:pass
else:raise AssertionError('Changed manifest accepted')
scope=json.loads((W/'reports/baseline-integration-20261026/tla_direct/SCOPE.json').read_text())
for r in scope:
 _,op,cid=r['key']
 assert v.matches(op,case(op,cid))==(r['key']==v.config()['key'])
# Exercise real deployment plumbing with an intercepted fake TileLang import only.
# Private AST loader + deployment + original adapter modules are genuinely loaded;
# they contain no top-level device import. No real compiler/library is loaded.
v._prepare=original_prepare
sys.modules['setuptools']=types.ModuleType('setuptools')
fake_tl=types.SimpleNamespace(__file__=v.config()['runtime']+'/tilelang/__init__.py',_LIB_PATH=v.config()['compiler']+'/handoff/lib/libtilelang_module.so')
original_import=builtins.__import__
def safe_import(name,*a,**kw):
 if name=='tilelang':return fake_tl
 return original_import(name,*a,**kw)
builtins.__import__=safe_import
os.environ['TLA_VERIFIED_WHERE_OUTPUT']=str(T/'mock-output')
try:
 n,tl,a,out=v._prepare()
 assert tl is fake_tl and n.runtime() is fake_tl
 assert pathlib.Path(a.__file__)==pathlib.Path(v.config()['private'])/'adapter.py'
 assert str(n.VENDOR)==v.config()['vendor']
 assert os.environ['TL_PLATFORM']=='A5'
 assert os.environ['TILELANG_LIBRARY_PATH']==v.config()['compiler']+'/handoff/lib'
 assert 'TVM_IMPORT_PYTHON_PATH' not in os.environ
 assert sys.path[0]==v.config()['runtime']
finally:builtins.__import__=original_import
assert not blocked
assert not any(name.split('.')[0] in ('torch','torch_npu','tilelang','tvm') for name in sys.modules)
result=dict(status='passed',device_imports=0,npu_runs=0,new_numerical_successes=0,all_provider_files_compile=True,old_dispatch_ast_preserved=True,existing_provider_count=len(tla.PROVIDERS),unknown_cases_old_routes_checked=len(tla.PROVIDERS),exact_scope_keys_checked=len(scope),verified_key_registry_mock=True,changed_manifest_rejected=True,real_deployment_setup_with_fake_runtime=True,pins_checked=len(v.config()['pins']),registered_count=1,verified_receipts_reused_count=1)
(T/'STATIC_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
