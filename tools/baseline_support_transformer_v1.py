"""Restore the existing clean ChunkKdaFwd deployment and top-level adapter."""
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / '.cache/all-pairs/full_rollout_20261024/local24/group3/recovery_1646/tasks/g3-linear-5b6c676bc2d2092f-r1622-r1646'
if os.environ.get('BASELINE_SUPPORT_TRANSFORMER_V1') != '1':
    env = dict(os.environ)
    env.update(json.loads((B / 'NATIVE_READY.json').read_text())['environment'])
    env['BASELINE_SUPPORT_TRANSFORMER_V1'] = '1'
    os.execve(sys.executable, [sys.executable, *sys.argv], env)
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT), str(B), str(ROOT / 'tools')]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes
from baseline_support_transformer_chunk_v1 import dispatch


def factory(case, library):
    if library != 'ops-transformer' or case._call.signature.name not in ('GDNFwdOp', 'KDAFwdOp', 'DeltaNetFwdOp', 'DeltaNetRecurrentFwdOp'):
        raise ValueError('Only original complete ChunkKdaFwd recurrence is owned')
    def build():
        candidate = dispatch(case)
        return dataclasses.replace(candidate, source_sha256={
            **candidate.source_sha256,
            **source_hashes(Path(__file__), ROOT / 'tools/baseline_support_transformer_chunk_v1.py', B / 'NATIVE_READY.json')})
    return build


if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
