import sys, pathlib, json, hashlib, types, importlib, importlib.abc, ctypes
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/transformer_dense'
IDS=['mha-decode-pv-float8_e4m3fn-bfloat16','mha-decode-pv-nk-float8_e4m3fn-bfloat16','mha-decode-qk-float8_e4m3fn-bfloat16','mha-decode-qk-nk-float8_e4m3fn-bfloat16','square-1k-nk-float8_e4m3fn-bfloat16']
def ref(p):
 p=pathlib.Path(p);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ['torch','torch_npu','acl','tvm','triton','tilelang']:raise RuntimeError('Blocked real device import '+fullname)
sys.meta_path.insert(0,Block())
def forbidden(*a,**kw):raise RuntimeError('Forbidden dynamic library load')
ctypes.CDLL=forbidden;ctypes.PyDLL=forbidden
for name in ['torch','benchmarks.api','benchmarks.ascend_baselines','benchmarks.ascend_math_more']:
 sys.modules[name]=types.ModuleType(name)
sys.modules['torch'].int8='int8';sys.modules['torch'].__audit_stub__=True
sys.modules['benchmarks.api'].Implementation=lambda run:types.SimpleNamespace(run=run)
sys.modules['benchmarks.ascend_baselines'].Candidate=lambda impl,api,adapt,hashes,**kw:dict(run_module=impl.run.__module__,run_name=impl.run.__name__,api=api,adaptation=adapt)
sys.modules['benchmarks.ascend_baselines'].source_hashes=lambda *files:{}
sys.modules['benchmarks.ascend_baselines'].UnsupportedCase=ValueError
sys.modules['benchmarks.ascend_baselines'].WORKSPACE=W
class Fn:pass
construct=[]
def stub(*a,**kw):
 construct.append({'api':a[0],'arg_count':len(a[1]),'host_library':str(kw['host_library'])})
 return types.SimpleNamespace(codes={},base=types.SimpleNamespace(aclCreateTensorList=Fn(),aclDestroyTensorList=Fn()))
sys.modules['benchmarks.ascend_math_more']._MathCall=stub
sys.path.insert(0,str(W/'TileOPs'))
registry=importlib.import_module('benchmarks.ascend_all_pairs.registry');routes=[];receipts=[]
for cid in IDS:
 cp=P/'fp8_siblings01'/('BmmFP8FwdOp__'+cid+'.json');c=json.loads(cp.read_text());case=types.SimpleNamespace(id=cid,params=c['case_snapshot']['parameters'],_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='BmmFP8FwdOp')))
 result=registry.candidate_factory(case,'ops-transformer')();mod=sys.modules[result['run_module']]
 routes.append(dict(case_id=cid,result=result,module=ref(mod.__file__),constructor=construct[-1]))
 receipts.append({'case_id':cid,'receipt':ref(cp),'source_sha256':c['source_sha256'],'adapter_sha256':c['adapter_sha256']})
files=[W/'TileOPs/benchmarks/ascend_math_more.py',W/'TileOPs/benchmarks/ascend_baselines.py',P/'run_fp8_siblings.py',P/'fp8_siblings01_guard.log',P/'fp8_siblings01_guard.exit',P/'fp8_siblings01/guard_snapshot.json']
# Select precise source and native candidates from the receipt's vendor, no log traversal.
source=receipts[0]['source_sha256'];vendor=P/'gemm/private_provider';selected=[]
for path,expected in source.items():
 if any(path.endswith(s) for s in ['libcust_opapi.so','dynamic/grouped_matmul.py','dynamic/grouped_matmul_apt.py','op_kernel/grouped_matmul_apt.cpp']):
  f=W/path;selected.append(dict(**ref(f),historical_expected=expected,matches=ref(f)['sha256']==expected,text=f.read_text() if f.suffix=='.py' else None))
# Inventory constrained to the known vendor kernel and this run's isolated cache.
artifacts=[]
for root in [vendor/'op_impl/ai_core/tbe/kernel/ascend950/grouped_matmul_apt',P/'fp8_siblings01/ASCEND_CACHE_PATH']:
 for f in root.rglob('*'):
  if f.is_file() and f.suffix in ['.o','.json']:
   r=ref(f);r['elf']=f.read_bytes()[:4]==b'\x7fELF';r['mtime_ns']=f.stat().st_mtime_ns
   if f.suffix=='.json':r['metadata']=json.loads(f.read_text())
   artifacts.append(r)
texts=[dict(**ref(f),text=f.read_text()) for f in files if f.suffix=='.py']
print(json.dumps(dict(routes=routes,registry=ref(registry.__file__),provider=ref(sys.modules['benchmarks.ascend_all_pairs.ops_transformer'].__file__),receipts=receipts,dependencies=[ref(f) for f in files],selected_sources=selected,artifacts=artifacts,texts=texts,real_device_imports=False,native_libraries_loaded=False)))
