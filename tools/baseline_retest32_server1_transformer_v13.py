"""Original recurrent provider; top-level public ACLNN binding for ARM deployment."""
import sys,types,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/old1076_transformer'))
name='benchmarks.ascend_all_pairs.ops_transformer'
package=types.ModuleType(name);package.__path__=[str(ROOT/'TileOPs/benchmarks/ascend_all_pairs/ops_transformer')];package.__package__=name
sys.modules[name]=package
from native_call import NativeCall
from benchmarks.ascend_all_pairs.ops_transformer import adapters,native
from benchmarks.ascend_baselines import Candidate,source_hashes
from benchmarks.api import Implementation
native.GraphCall=NativeCall

def candidate(binding,run,op,directory,adaptation,host=False):
 files=[Path(__file__),Path(adapters.__file__),ROOT/'tools/old1076_transformer/native_call.py']
 files.extend(p for p in binding.vendor.rglob('*') if p.is_file())
 files.extend(p for p in (ROOT/'ascend_baselines/ops-transformer'/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.hpp'))
 return Candidate(Implementation(run),'ops-transformer::'+op+' via public ACLNN',adaptation+' Original public ACLNN binding for ARM host.',source_hashes(*files),timing_mode='host_roundtrip',cleanup=binding.cleanup)
adapters._candidate=candidate
package.PROVIDERS={'DeltaNetRecurrentFwdOp':adapters.recurrent_family}
from benchmarks.ascend_all_pairs import registry
original_factory=registry.candidate_factory

def factory(case,library):
 if library=='ops-transformer':
  assert case._call.signature.name=='DeltaNetRecurrentFwdOp'
  return lambda: adapters.recurrent_family(case)
 return original_factory(case,library)
registry.candidate_factory=factory
