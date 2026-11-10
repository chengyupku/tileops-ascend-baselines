import sys,os,json,importlib.util,runpy
from pathlib import Path
ROOT=Path('/opt/tileops-baseline-cache/workspace');HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT),str(ROOT/'tools')]
task=json.loads(Path(sys.argv[sys.argv.index('--task')+1]).read_text());lib=task['key'][0];kind={'tilelang-ascend':'tla','vllm-ascend':'vllm','catlass':'catlass','ops-tensor':'ops_tensor'}[lib]
import benchmarks.ascend_all_pairs
name='benchmarks.ascend_all_pairs.'+kind
spec=importlib.util.spec_from_file_location(name,HERE/'providers'/kind/'__init__.py',submodule_search_locations=[str(HERE/'providers'/kind)])
m=importlib.util.module_from_spec(spec);sys.modules[name]=m;setattr(benchmarks.ascend_all_pairs,kind,m);
if kind=='tla':
 import importlib,functools
 native=importlib.import_module(name+'.native')
 adapters=importlib.import_module(name+'.adapters')
 def make(op,case):
  native.prepare_deployment();return adapters.factory(op,case)
 m.PROVIDERS={op:functools.partial(make,op) for op in ('MulFwdOp','AddFwdOp')}
else:spec.loader.exec_module(m)
if kind in ('catlass','ops_tensor'):
 import importlib
 importlib.import_module(name+'.build')
for n,mod in list(sys.modules.items()):
 if n.startswith(name+'.'):
  if hasattr(mod,'WORKSPACE'):mod.WORKSPACE=ROOT
  if hasattr(mod,'VENDOR'):mod.VENDOR=ROOT/'ascend_baselines'/lib
  if hasattr(mod,'CACHE'):mod.CACHE=ROOT/'.cache/all-pairs'/kind
if kind=='tla':
 sys.path.insert(0,str(ROOT/'.cache/all-pairs/remote_tla_compiler/python'));sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
 import setuptools
 from benchmarks.ascend_all_pairs.tla.native import runtime
 tl=runtime();out=Path(sys.argv[sys.argv.index('--out')+1]);cache=out/'jit_kernel_cache';cache.mkdir();tl.set_cache_dir(str(cache))
runpy.run_path(str(HERE/'bootstrap.py'),run_name='__main__')
