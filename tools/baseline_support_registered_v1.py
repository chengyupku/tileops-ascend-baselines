"""Run declared original API candidates without historical admission removals."""
import dataclasses
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes
original = registry.candidate_factory


def factory(case, library):
    op = case._call.signature.name
    def build():
        make = None
        if library == 'ops-transformer':
            from benchmarks.ascend_all_pairs.ops_transformer import adapters
            make = {'LayerNormFwdOp': adapters.layernorm,
                    'GroupNormFwdOp': adapters.channel_norm,
                    'DeltaNetChunkFwdOp': adapters.chunk_delta}.get(op)
        elif library == 'TileKernels':
            from benchmarks.ascend_all_pairs.tilekernels import DIAGNOSTIC_PROVIDERS
            make = DIAGNOSTIC_PROVIDERS.get(op)
        candidate = make(case) if make else original(case, library)()
        return dataclasses.replace(candidate, source_sha256={
            **candidate.source_sha256, **source_hashes(Path(__file__))})
    return build


if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
