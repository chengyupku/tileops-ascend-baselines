"""Final bounded SGL GDN attempt: original FP16 QKV and FP32 gate/beta."""
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
            return native.run(q,k,v,g.float(),beta.float(),*rest)
        return dataclasses.replace(candidate,
            implementation=dataclasses.replace(native,run=run),
            adaptation=candidate.adaptation+' Lossless FP32 gate/beta transport with native FP16 QKV; all casts timed.',
            source_sha256={**candidate.source_sha256,**source_hashes(Path(__file__))})
    return build


if __name__ == '__main__':
    registry.candidate_factory=factory
    runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
