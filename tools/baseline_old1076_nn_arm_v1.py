"""Top-level binding to the existing hash-verified original ARM NN vendor."""
import dataclasses,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_nn import staging
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
 if library!='ops-nn' or case._call.signature.name not in ('BmmFwdOp','MulFwdOp'):raise ValueError('Only verified ARM vendor APIs in pilot scope')
 def build():
  c=staging.CANDIDATES[case._call.signature.name](case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
