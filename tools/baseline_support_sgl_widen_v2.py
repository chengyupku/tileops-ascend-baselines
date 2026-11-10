"""Bounded second attempt: widen public-API inputs, preserve original kernels."""
import dataclasses
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.sgl.linear import _triton_gdn
from benchmarks.ascend_baselines import source_hashes


def factory(case, library):
    if library != 'sgl-kernel-npu' or case._call.signature.name != 'GDNFwdOp':
        raise ValueError('Only SGL GDN is owned')
    def build():
        candidate = _triton_gdn(case)
        native = candidate.implementation
        def run(q,k,v,g,beta,*rest):
            out,state = native.run(q.float(),k.float(),v.float(),g.float(),beta.float(),*rest)
            return out.to(v.dtype),state
        return dataclasses.replace(candidate,
            implementation=dataclasses.replace(native,run=run),
            adaptation=candidate.adaptation+' Lossless FP32 input widening through original public API, output dtype restoration; all casts timed.',
            source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))})
    return build


if __name__ == '__main__':
    registry.candidate_factory=factory
    runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
