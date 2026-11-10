"""Original W4A16 provider and dynamic INT4 descriptors in private ARM environment."""
import sys,types,dataclasses,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name
sys.modules[name]=package
path=ROOT/'reports/baseline-retest32-20261030/server1/deployments/w4a16_frontend/native.py'
spec=importlib.util.spec_from_file_location(name+'.native',path)
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
native.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_cache/root'
from benchmarks.ascend_all_pairs.ops_transformer import adapters
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_normrope_v47/root'
import benchmarks.ascend_transformer_graph as original
original_candidate=original._candidate
from benchmarks.ascend_baselines import source_hashes

def candidate(binding,run,op,directory,adaptation,host=False):
 c=original_candidate(binding,run,op,directory,adaptation)
 def bounded_run(*args,**kwargs):
  try:return run(*args,**kwargs)
  finally:binding.cleanup()
 files=[Path(__file__),path,Path(adapters.__file__),Path(adapters.__file__).with_name('graph_layout.py'),Path(adapters.__file__).with_name('list_tensor_arg.py')]
 return dataclasses.replace(c,implementation=dataclasses.replace(c.implementation,run=bounded_run),timing_mode='host_roundtrip',source_sha256={**c.source_sha256,**source_hashes(*files)})
adapters._candidate=candidate
original._candidate=candidate
package.PROVIDERS={'PagedKVCacheWriteFwdOp':adapters.cache_write,'MLAKVCacheWriteFwdOp':adapters.mla_cache_write,'PagedKVCacheGatherFwdOp':adapters.cache_gather}
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory
def factory(case,library):
 if library=='ops-transformer':
  assert case._call.signature.name in package.PROVIDERS
  return lambda:package.PROVIDERS[case._call.signature.name](case)
 return original_factory(case,library)
registry.candidate_factory=factory
