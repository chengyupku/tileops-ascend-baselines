from pathlib import Path
import importlib.util,importlib.abc,sys,types,json,hashlib
sys.dont_write_bytecode=True
class BlockRuntime(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","torch_npu","tilelang","triton"):raise ImportError("CPU registration check forbids device runtime: "+fullname)
sys.meta_path.insert(0,BlockRuntime())
W=Path(__file__).resolve().parent;R=W.parents[2];P=R/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer'
# stdlib-only stub parents avoid importing any runtime from unrelated benchmark setup.
for name,path in [('benchmarks',R/'TileOPs/benchmarks'),('benchmarks.ascend_all_pairs',R/'TileOPs/benchmarks/ascend_all_pairs')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
spec=importlib.util.spec_from_file_location('benchmarks.ascend_all_pairs.ops_transformer',P/'__init__.py',submodule_search_locations=[str(P)]);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
name=spec.name+'.integration700_transformer_linear_gdn';own=sys.modules[name];original=own.build;own.build=lambda c:('own',c.id)
keys=json.loads((W/'REGISTERED.json').read_text());assert all(mod.PROVIDERS['GDNFwdOp'](types.SimpleNamespace(id=i))==('own',i) for i in keys)
fallback=own.wrap(lambda c:('prior',c.id));assert fallback(types.SimpleNamespace(id='outside_scope'))==('prior','outside_scope')
assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
own.build=original
(W/'STATIC_REGISTRATION.json').write_text(json.dumps({'keys_checked':keys,'exact_dispatch':True,'fallback_preserved':True,'no_torch_import':True,'module_sha256':hashlib.sha256((P/'integration700_transformer_linear_gdn.py').read_bytes()).hexdigest()},indent=2));print('static routing checked',len(keys))
