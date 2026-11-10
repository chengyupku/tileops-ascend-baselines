"""Use original BMM graph provider with a private unchanged ARM vendor."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from benchmarks.ascend_all_pairs import registry
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-retest32-20261030/server1/deployments/ops_graph/root'
name='benchmarks.ascend_all_pairs.ops_transformer';package=types.ModuleType(name)
package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_transformer.adapters import bmm,grouped_gemm,pointwise_conv
from benchmarks.ascend_all_pairs.ops_transformer.conv import conv2d
from benchmarks.ascend_baselines import source_hashes
from benchmarks.ascend_all_pairs.ops_transformer import adapters
from benchmarks.ascend_transformer_graph import _candidate as original_candidate

def graph_candidate(binding,run,op,directory,adaptation,host=False):
 # Actual graph route has no infer-shim dependency; fingerprint its real source.
 c=original_candidate(binding,run,op,directory,adaptation)
 return dataclasses.replace(c,timing_mode='host_roundtrip' if host else c.timing_mode,source_sha256={**c.source_sha256,**source_hashes(Path(adapters.__file__),Path(__file__))})
adapters._candidate=graph_candidate
original_factory=registry.candidate_factory

def factory(case,library):
 if library!='ops-transformer':return original_factory(case,library)
 assert case._call.signature.name in ('BmmFwdOp','GroupedGemmFwdOp','Conv1dFwdOp','Conv3dFwdOp','Conv2dFwdOp')
 def build():
  c={'BmmFwdOp':bmm,'GroupedGemmFwdOp':grouped_gemm,'Conv1dFwdOp':pointwise_conv,'Conv3dFwdOp':pointwise_conv,'Conv2dFwdOp':conv2d}[case._call.signature.name](case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
