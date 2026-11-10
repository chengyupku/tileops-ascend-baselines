"""Original routing Add via original public ACLNN ABI; no implementation edits."""
import ctypes as C
import dataclasses,sys,types,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name;sys.modules[name]=package
from benchmarks.ascend_all_pairs.ops_transformer import native,adapters,routing_arithmetic
from benchmarks.ascend_math_more import _MathCall
from benchmarks.ascend_baselines import Candidate,source_hashes
from benchmarks.api import Implementation
class RoutingCall:
    def __init__(self,library,vendor_name=None):
        assert library=='ops-transformer'
        self.vendor=Path(os.environ['OLD1076_TRANSFORMER_VENDOR'])
        self.binding=_MathCall('aclnnMoeFinalizeRoutingV2',[C.c_void_p]*7+[C.c_int64,C.c_void_p],library=library,host_library=self.vendor/'op_api/lib/libcust_opapi.so')
    def call(self,op,inputs,outputs,attrs):
        assert op=='MoeFinalizeRoutingV2' and len(inputs)==7 and len(outputs)==1
        assert attrs=={'drop_pad_mode':0,'k':1}
        b=self.binding;record=b.record()
        args=[None if t is None else b.tensor(t,record) for t in inputs]
        args += [attrs['drop_pad_mode'],b.tensor(outputs[0],record)]
        b.call(args,record,inputs[0].device)
    def cleanup(self):self.binding.cleanup()
native.GraphCall=RoutingCall
def candidate(binding,run,op,directory,adaptation,host=False):
    paths=[Path(__file__),Path(adapters.__file__),Path(routing_arithmetic.__file__),ROOT/'TileOPs/benchmarks/ascend_math_more.py']
    paths.extend(p for p in binding.vendor.rglob('*') if p.is_file())
    paths.extend(p for p in (ROOT/'ascend_baselines/ops-transformer'/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.hpp'))
    return Candidate(Implementation(run),'ops-transformer::'+op+' via original ACLNN',adaptation+' Original public ACLNN ARM binding.',source_hashes(*paths),timing_mode='host_roundtrip',cleanup=binding.cleanup)
adapters._candidate=candidate
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory
def factory(case,library):
    if library!='ops-transformer':return original_factory(case,library)
    assert case._call.signature.name=='AddFwdOp' and case.params['input'][1] in ('float32','bfloat16')
    return lambda:routing_arithmetic.dispatch(case)
registry.candidate_factory=factory
