import importlib,sys,json
from pathlib import Path
sys.dont_write_bytecode=True
W=Path('/root/dyq/workspace');sys.path.insert(0,str(W/'TileOPs'))
class NoDevice:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','acl','tilelang'):raise AssertionError(name)
sys.meta_path.insert(0,NoDevice())
p='benchmarks.ascend_all_pairs.ops_nn'
m=importlib.import_module(p+'.verified_runtime')
for r in m.RECORDS.values():
 mod,fn=r['factory'].split(':'); assert callable(getattr(importlib.import_module(p+'.verified_20261026.'+mod),fn))
assert not any(x in sys.modules for x in ('torch','torch_npu'))
print('All 16 concrete factory attributes import without torch, torch_npu, ACL or tilelang.')
