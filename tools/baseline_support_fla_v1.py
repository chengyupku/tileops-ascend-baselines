"""Restore lossless gate/beta FP32 transport to the unchanged native FLA API."""
import dataclasses
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.fla.providers import gdn
from benchmarks.ascend_baselines import source_hashes


def factory(case, library):
    if (library, case._call.signature.name) != ('flash-linear-attention-npu', 'GDNFwdOp'):
        raise ValueError('Only original FLA GDN is owned by this entrypoint')

    def build():
        candidate = gdn(case)
        native = candidate.implementation

        def run(q, k, v, g, beta, *rest):
            return native.run(q, k, v, g.float(), beta.float(), *rest)

        return dataclasses.replace(candidate,
            implementation=dataclasses.replace(native, run=run),
            adaptation=candidate.adaptation + ' Lossless gate/beta FP32 transport, inside timed call.',
            source_sha256={**candidate.source_sha256, **source_hashes(Path(__file__))})
    return build


if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
