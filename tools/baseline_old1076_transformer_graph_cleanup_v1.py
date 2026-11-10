"""Private original graph API binding with synchronous resource release."""
import dataclasses
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_transformer import native, adapters, topk
from benchmarks.ascend_baselines import source_hashes
OriginalGraphCall=native.GraphCall
class ReleasedGraphCall(OriginalGraphCall):
    def call(self,*args,**kwargs):
        result=super().call(*args,**kwargs)
        self.cleanup()
        return result
native.GraphCall=ReleasedGraphCall
def factory(case,library):
    assert library=='ops-transformer'
    def build():
        op=case._call.signature.name
        provider={'TopKSelectFwdOp':topk.topk_select,'HadamardTransformFwdOp':adapters.hadamard}[op]
        candidate=provider(case)
        return dataclasses.replace(candidate,source_sha256=candidate.source_sha256|source_hashes(Path(__file__)),adaptation=candidate.adaptation+' Release descriptors/workspace via original synchronized cleanup after each complete public call; no kernel or compiler changes.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
