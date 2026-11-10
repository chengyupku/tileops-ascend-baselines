"""Expose existing original ops-math candidates to the unchanged full checker.

The checker already uses the original CPU reference for ops-math. Loading the
original graph OPP in this foreground process lets the unified profiler observe
the complete call, rather than losing kernels in a subprocess.
"""
import dataclasses
import os
from pathlib import Path
import runpy
import sys
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('BASELINE_SUPPORT_MATH_V1') != '1':
    env = dict(os.environ)
    vendor = ROOT / '.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
    env.update(BASELINE_SUPPORT_MATH_V1='1', ASCEND_CUSTOM_OPP_PATH=str(vendor),
               FLA_NPU_DISABLE_PTH='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env['LD_LIBRARY_PATH'] = str(vendor / 'op_api/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
from benchmarks.ascend_all_pairs.ops_math.composition_specs import COMPOSITIONS
from benchmarks.ascend_all_pairs.ops_math.composition import make as composition
from benchmarks.ascend_baselines import source_hashes, UnsupportedCase

providers = dict(NATIVE_CANDIDATES)
for name in COMPOSITIONS:
    providers.setdefault(name, partial(composition, op=name))


def factory(case, library):
    if library != 'ops-math':
        raise ValueError('This versioned entrypoint only owns ops-math')
    op = case._call.signature.name

    def build():
        if op not in providers:
            raise UnsupportedCase('No existing top-level ops-math candidate: ' + op)
        candidate = providers[op](case)
        return dataclasses.replace(candidate, source_sha256={
            **candidate.source_sha256, **source_hashes(Path(__file__))})
    return build


if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
