"""Original in-process CATLASS vector calls; require complete native profiler capture."""
import dataclasses
import json
import shutil
import time
import hashlib
import math
from pathlib import Path
import runpy
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE),str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(HERE.with_name('arithmetic9_20261027'))]
from tools import measure_baselines as timing
from original_catlass.vector import vector

current = None
original_parser = timing._parse_ffts_profile
def checked_parser(prof_path, *args, **kwargs):
    result = original_parser(prof_path, *args, **kwargs)
    if current and not kwargs.get('require_marker_seen', False):
        from profile_task_identity import verify_capture
        try:
            report = verify_capture(prof_path, current['entrypoint'], current['launches'])
            assert result['profile_kernel_count'] == report['all_device_records']
            assert result.get('profile_excluded_kernel_count', 0) == 0
        except Exception:
            destination = HERE / 'baseline' / ('catlass-failed-capture-' + str(time.time_ns()))
            destination.mkdir(parents=True)
            for path in prof_path.rglob('*'):
                if path.is_file() and any(key in path.name for key in ('ffts_profile', 'task_track', 'hash_dic')):
                    target = destination / path.relative_to(prof_path)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
            raise
        report.update(case_id=current['case_id'], op=current['op'], latency_ms=result['latency_ms'])
        with (HERE / 'baseline' / ('catlass-capture-checks-' + str(__import__('os').getpid()) + '.jsonl')).open('a') as output:
            output.write(json.dumps(report) + '\n')
    return result
timing._parse_ffts_profile = checked_parser

def make(case):
    global current
    candidate = vector(case, op=case._call.signature.name)
    run = candidate.implementation.run
    closure = dict(zip(run.__code__.co_freevars, (cell.cell_contents for cell in run.__closure__)))
    artifact = closure['artifact']
    current = {'case_id':case.id, 'op':case._call.signature.name, 'entrypoint': artifact.entrypoint,
               'launches': math.ceil(math.prod(case.params['output'][0]) / 1024)}
    files = [Path(__file__), HERE/'profile_task_identity.py', HERE.with_name('arithmetic9_20261027') / 'original_catlass/build.py', HERE.with_name('arithmetic9_20261027') / 'original_catlass/dsl_deployment.py']
    # The original generic adapter labels host_roundtrip. This dedicated route
    # contains only in-process native launches, so apply the project device-kernel
    # metric and verify every native launch appears in every profiler window.
    # Python loop overhead remains excluded, exactly as for all other opponents.
    return dataclasses.replace(candidate, timing_mode='device_async',
        adaptation=candidate.adaptation + ' Dedicated msprof device-kernel-sum measurement; original host_roundtrip label is preserved here as provenance. Every native launch must be captured; host dispatch time is not included.',
        source_sha256=candidate.source_sha256 | {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

from benchmarks.ascend_all_pairs import registry
prior = registry.candidate_factory
registry.candidate_factory = lambda case, library: (lambda: make(case)) if library == 'catlass' else prior(case, library)
runpy.run_path(str(HERE / 'baselines.py'), run_name='__main__')
