"""Validate original staged NN candidates under the unchanged full checker."""
import dataclasses,runpy,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
import ctypes,os
vendor=Path(os.environ['OPS_NN_VENDOR'])
ctypes.CDLL('/usr/local/Ascend/cann-9.1.0/lib64/libes_nn.so',mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(str(vendor/'op_proto/es/lib/linux/aarch64/libes_nn.so'),mode=ctypes.RTLD_GLOBAL)
from benchmarks.ascend_all_pairs import registry
name='benchmarks.ascend_all_pairs.ops_nn'
package=types.ModuleType(name);package.__path__=[str(ROOT/'reports/baseline-old1076-20261029/arm_nn_adapters')];package.__package__=name
sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_nn import staging
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
 if library!='ops-nn':raise ValueError('ops-nn staging only')
 def build():
  c=staging.CANDIDATES[case._call.signature.name](case)
  return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
 return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
