"""Original isolated ops-math correctness; do not invent parent-only timings.

Use this until profiling covers the native foreground child. Verified numerical
support and a reusable performance cache remain separate recorded facts.
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
        command = [str(ROOT / 'tools/baseline_support_math_child_v2.py')
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
            **source_hashes(Path(__file__), ROOT / 'tools/baseline_support_math_child_v2.py')})
    return build


def unavailable_parent_profile(*args, **kwargs):
    raise RuntimeError('Full original checker passed. Original native child requires child-profiler coverage; parent-only timing deliberately not recorded.')


if __name__ == '__main__':
    registry.candidate_factory = factory
    from tools import measure_baselines
    measure_baselines._time_reference = unavailable_parent_profile
    runpy.run_path(str(ROOT / 'tools/measure_baseline_cache_pair.py'), run_name='__main__')
