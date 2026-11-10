"""Original isolated correctness plus full native-child unified profiling.

Original full checker stays in its clean environment. The native foreground
child runs the unchanged unified timing core, with full source/guard binding.
"""
import dataclasses
from functools import partial
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.ops_math import isolated
from benchmarks.ascend_baselines import source_hashes
import subprocess

original_run = subprocess.run


def child_run(command, *args, **kwargs):
    if isinstance(command, list) and any(str(x).endswith('/ops_math/native_child.py') for x in command):
        command = [str(ROOT / 'tools/baseline_support_math_child_v6.py')
                   if str(x).endswith('/ops_math/native_child.py') else x for x in command]
    return original_run(command, *args, **kwargs)


class SubprocessProxy:
    run = staticmethod(child_run)
    def __getattr__(self, name):
        return getattr(subprocess, name)


isolated.subprocess = SubprocessProxy()


def factory(case, library):
    if library != 'ops-math':
        raise ValueError('Only ops-math is owned by this entrypoint')
    def build():
        candidate = isolated.make(case, case._call.signature.name)
        return dataclasses.replace(candidate, source_sha256={
            **candidate.source_sha256,
            **source_hashes(Path(__file__), ROOT / 'tools/baseline_support_math_child_v6.py', ROOT / 'tools/baseline_support_math_profile_child_v6.py', ROOT / 'tools/measure_baseline_cache_math_child_v6.py', ROOT / 'tools/baseline_support_math_logical_v6.py')})
    return build



if __name__ == '__main__':
    registry.candidate_factory = factory
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_math_child_v6.py', ROOT / 'tools/baseline_support_math_logical_v6.py'), run_name='__main__')
