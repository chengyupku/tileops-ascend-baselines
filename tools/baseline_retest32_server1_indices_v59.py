"""Original index providers with isolated original ARM graph/ACLNN bindings."""
import sys,types,dataclasses,ctypes as C
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'reports/baseline-retest32-20261030/server1/deployments'
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
import benchmarks.ascend_graph_call as graph
graph.WORKSPACE=H/'ops_attention_v51/root'
from benchmarks.ascend_all_pairs.ops_transformer import native,adapters,arg_indices,topk,routing_arithmetic
from benchmarks.ascend_math_more import _MathCall
from benchmarks.ascend_baselines import Candidate,source_hashes
from benchmarks.api import Implementation
class RoutingCall:
 def __init__(self):
  self.vendor=H/'ops_graph/vendors/allpairs_transformer'
  self.binding=_MathCall('aclnnMoeFinalizeRoutingV2',[C.c_void_p]*7+[C.c_int64,C.c_void_p],library='ops-transformer',host_library=self.vendor/'op_api/lib/libcust_opapi.so')
 def call(self,op,inputs,outputs,attrs):
  assert op=='MoeFinalizeRoutingV2' and len(inputs)==7 and len(outputs)==1
  assert attrs=={'drop_pad_mode':0,'k':1}
  b=self.binding;record=b.record();args=[None if t is None else b.tensor(t,record) for t in inputs];args += [attrs['drop_pad_mode'],b.tensor(outputs[0],record)];b.call(args,record,inputs[0].device)
 def cleanup(self):self.binding.cleanup()
class GraphBinding:
 def __init__(self,root):
  previous=graph.WORKSPACE
  try:
   graph.WORKSPACE=root
   self.binding=graph.GraphCall('ops-transformer')
  finally:graph.WORKSPACE=previous
  self.vendor=self.binding.vendor
 def call(self,*args,**kwargs):return self.binding.call(*args,**kwargs)
 def cleanup(self):self.binding.cleanup()
def index_binding(library,vendor_name=None):
 assert library=='ops-transformer'
 if vendor_name=='selectionpairs_transformer':return GraphBinding(H/'ops_selection/root')
 assert vendor_name is None
 return RoutingCall()
def topk_binding(library,vendor_name=None):
 assert library=='ops-transformer' and vendor_name is None
 return GraphBinding(H/'ops_graph/root')
def candidate(binding,run,op,directory,adaptation,host=False):
 paths=[Path(__file__),Path(adapters.__file__),Path(arg_indices.__file__),Path(topk.__file__),Path(routing_arithmetic.__file__),Path(graph.__file__),ROOT/'TileOPs/benchmarks/ascend_math_more.py']
 paths.extend(p for p in binding.vendor.rglob('*') if p.is_file())
 paths.extend(p for p in (ROOT/'ascend_baselines/ops-transformer'/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.hpp'))
 def bounded_run(*args,**kwargs):
  try:return run(*args,**kwargs)
  finally:binding.cleanup()
 return Candidate(Implementation(bounded_run),'ops-transformer::'+op+' via original API',adaptation+' Original ARM API bindings with isolated vendor paths.',source_hashes(*paths),timing_mode='host_roundtrip',cleanup=binding.cleanup)
adapters._candidate=candidate
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory
def factory(case,library):
 if library!='ops-transformer':return original_factory(case,library)
 op=case._call.signature.name
 assert op in ('ArgminFwdOp','TopKSelectFwdOp')
 def build():
  native.GraphCall=index_binding if op=='ArgminFwdOp' else topk_binding
  return arg_indices.arg_indices(case) if op=='ArgminFwdOp' else topk.topk_select(case)
 return build
registry.candidate_factory=factory
