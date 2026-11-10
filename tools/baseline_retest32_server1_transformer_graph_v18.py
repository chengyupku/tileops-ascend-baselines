"""Use original BMM graph provider with a private unchanged ARM vendor."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from benchmarks.ascend_all_pairs import registry
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_graph/root'
name='benchmarks.ascend_all_pairs.ops_transformer';package=types.ModuleType(name)
package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_transformer.adapters import bmm
from benchmarks.ascend_baselines import source_hashes
original_factory=registry.candidate_factory

def factory(case,library):
 if library!='ops-transformer':return original_factory(case,library)
 assert case._call.signature.name=='BmmFwdOp'
 def build():
  c=bmm(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
