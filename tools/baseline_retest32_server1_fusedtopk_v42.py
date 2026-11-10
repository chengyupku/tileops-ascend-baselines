"""Original recurrent provider; top-level public ACLNN binding for ARM deployment."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/old1076_transformer'))
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name
sys.modules[name]=package
import os
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=Path(os.environ['SERVER1_FUSED_GRAPH_ROOT'])
class NativeCall:
 def __init__(self,library,vendor_name=None):
  assert library=='ops-transformer'
  self.vendor=Path(os.environ['OLD1076_TRANSFORMER_VENDOR'])
  self.binding=graph.GraphCall(library)
 def call(self,op,inputs,outputs,attrs):
  assert op in ('MoeGatingTopKSoftmax','MoeGatingTopK','MoeFusedTopk')
  self.binding.call(op,inputs,outputs,attrs)
 def cleanup(self):self.binding.cleanup()
from benchmarks.ascend_all_pairs.ops_transformer import adapters,native,gating_normalization,routing
from benchmarks.ascend_baselines import Candidate,source_hashes
from benchmarks.api import Implementation
native.GraphCall=NativeCall

def candidate(binding,run,op,directory,adaptation,host=False):
 files=[Path(__file__),Path(adapters.__file__),Path(native.__file__),Path(gating_normalization.__file__),Path(routing.__file__),Path(graph.__file__),ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer/aclnn_bridge.py',ROOT/'TileOPs/benchmarks/ascend_math_more.py']
 files.extend(p for p in binding.vendor.rglob('*') if p.is_file())
 files.extend(p for p in (ROOT/'ascend_baselines/ops-transformer'/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.hpp'))
 def bounded_run(*args,**kwargs):
  try:return run(*args,**kwargs)
  finally:binding.cleanup()
 return Candidate(Implementation(bounded_run),'ops-transformer::'+op+' via original GraphCall',adaptation+' Original graph API binding for ARM host.',source_hashes(*files),timing_mode='host_roundtrip',cleanup=binding.cleanup)
adapters._candidate=candidate
package.PROVIDERS={'FusedTopKFwdOp':gating_normalization.fused_topk_dispatch}
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory

def factory(case,library):
 if library=='ops-transformer':
  assert case._call.signature.name in package.PROVIDERS
  return lambda: gating_normalization.fused_topk_dispatch(case)
 return original_factory(case,library)
registry.candidate_factory=factory
