import sys,runpy,dataclasses,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=ROOT/'reports/baseline-old1076-20261029/arm_graph_root'
name='benchmarks.ascend_all_pairs.ops_transformer';package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_transformer.adapters import bmm,grouped_gemm
from benchmarks.ascend_baselines import source_hashes
def factory(case,library):
 if library!='ops-transformer' or case._call.signature.name not in ('BmmFwdOp','GroupedGemmFwdOp','MoEGroupedGemmFwdOp'):raise ValueError('BMM pilot only')
 def build():
  c=(bmm if case._call.signature.name=="BmmFwdOp" else grouped_gemm)(case);return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
