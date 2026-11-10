import sys, pathlib, json, types, importlib.abc, traceback, importlib.util
sys.dont_write_bytecode=True
W=pathlib.Path('/opt/tileops-all-pairs/workspace');P=W/'reports/baseline-integration-700-20261026/tilelang_register5'
sys.path.insert(0,str(W/'TileOPs'))
class DeviceImportBlocked(BaseException):pass
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','torch_npu','tvm'}:
   raise DeviceImportBlocked(fullname)
  # Runtime location may be discovered but is never imported in this CPU test.
sys.meta_path.insert(0,Block())
# Stub the device package itself as a hard error for any accidental access.
stub=types.ModuleType('torch_npu')
def forbidden(name):raise DeviceImportBlocked('torch_npu.'+name)
stub.__getattr__=forbidden;sys.modules['torch_npu']=stub
from benchmarks.ascend_all_pairs.registry import candidate_factory
from benchmarks.ascend_all_pairs import tilelang as provider
from benchmarks.ascend_all_pairs.tilelang import integration700_tilelang_register5 as module
rows=[]
for op,cid in sorted(module.KEYS):
 case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name=op)),id=cid)
 try:candidate_factory(case,'tilelang')()
 except DeviceImportBlocked as e:
  stack=traceback.extract_tb(e.__traceback__)
  files=[x.filename for x in stack]
  assert str(P/'attempt02/frozen/impl.py') in files,files
  assert any(x.endswith('/integration700_tilelang_register5.py') for x in files)
  assert str(e)=='torch',str(e)
  rows.append(dict(key=['tilelang',op,cid],actual_registry=True,reached_frozen_factory=True,blocked_import=str(e),frames=[dict(file=x.filename,line=x.lineno,function=x.name) for x in stack]))
 else:raise AssertionError('unexpected device factory completion')
case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='FP8QuantFwdOp')),id='outside-scope')
try:candidate_factory(case,'tilelang')()
except NotImplementedError:pass
else:raise AssertionError('outside key accepted')
assert provider.PROVIDERS['BmmFwdOp'] is provider.bmm
assert 'torch' not in sys.modules and 'tilelang' not in sys.modules and 'tvm' not in sys.modules
result=dict(rows=rows,pins=module.inspect_pins(),outside_scope_rejected=True,existing_bmm_preserved=True,no_device_import=True,stub='torch_npu hard-block stub; torch and tvm blocked by MetaPathFinder before loader')
(P/'attempt02/REGISTRY_CPU.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
