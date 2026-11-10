"""Top-level binding to the existing hash-verified original ARM NN vendor."""
import dataclasses,runpy,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
# Load original submodules without unrelated historical registration side effects.
name='benchmarks.ascend_all_pairs.ops_nn'
package=types.ModuleType(name)
package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_nn')]
package.__package__=name
sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_nn.matrix import bmm
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
 if library!='ops-nn' or case._call.signature.name != 'BmmFwdOp':raise ValueError('Only verified ARM vendor APIs in pilot scope')
 def build():
  c=bmm(case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
