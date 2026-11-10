"""Lossless gate/beta transport through unchanged complete native KDA API."""
import dataclasses
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.vllm.linear_attention import linear_attention
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
    if library!='vllm-ascend' or case._call.signature.name!='GDNFwdOp':raise ValueError('Only vllm GDN')
    def build():
        c=linear_attention(case);impl=c.implementation
        def run(q,k,v,g,beta,*rest):return impl.run(q,k,v,g.float(),beta.float(),*rest)
        return dataclasses.replace(c,implementation=dataclasses.replace(impl,run=run),
            source_sha256={**c.source_sha256,**source_hashes(Path(__file__))},
            adaptation=c.adaptation+' Gate and beta losslessly widen to FP32 at original complete API; transforms timed.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
