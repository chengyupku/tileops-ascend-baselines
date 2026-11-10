"""Original PrePermute provider, isolated original ARM graph environment."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name
sys.modules[name]=package
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_normrope_v47/root'
import benchmarks.ascend_transformer_graph as original
from benchmarks.ascend_baselines import source_hashes
original_candidate=original._candidate
def candidate(binding,run,op,directory,adaptation):
 c=original_candidate(binding,run,op,directory,adaptation)
 def bounded_run(*args,**kwargs):
  try:return run(*args,**kwargs)
  finally:binding.cleanup()
 return dataclasses.replace(c,implementation=dataclasses.replace(c.implementation,run=bounded_run),timing_mode='host_roundtrip',source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
original._candidate=candidate
from benchmarks.ascend_rms_cases import transformer_rms_cases
from benchmarks.ascend_rope_cases import transformer_rope_cases
package.PROVIDERS={'RMSNormFwdOp':transformer_rms_cases,'RoPEFwdOp':transformer_rope_cases}
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory
def factory(case,library):
 if library=='ops-transformer':
  assert case._call.signature.name in package.PROVIDERS
  return lambda:package.PROVIDERS[case._call.signature.name](case)
 return original_factory(case,library)
registry.candidate_factory=factory
