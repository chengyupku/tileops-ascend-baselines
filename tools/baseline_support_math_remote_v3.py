"""Use the existing remote ops-math deployment with clean verifier startup."""
import dataclasses
import json
import os
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
state = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
if (state.get('phase') != 'running' or state.get('child',{}).get('pid') != os.getpgrp()
        or not state.get('owner','').startswith('baseline-cache-20261027/server2/')):
    raise RuntimeError('Actual cache foreground guard owner/group required')
# This is the current guard owner, not a forged historical receipt identity.
os.environ['OPS_MATH_GUARD_OWNER'] = state['owner']
if os.environ.get('SUPPORT_REMOTE_MATH_READY') != '1':
    env = dict(os.environ)
    env.update(SUPPORT_REMOTE_MATH_READY='1', ASCEND_CUSTOM_OPP_PATH='', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env['LD_LIBRARY_PATH'] = ':'.join(p for p in env.get('LD_LIBRARY_PATH','').split(':')
                                    if '/remote_ops_math/' not in p)
    os.execve(sys.executable, [sys.executable,*sys.argv], env)
sys.path[:0] = [str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from tools import baseline_support_math_isolated_v3 as isolated
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import source_hashes

def factory(case, library):
    make = isolated.factory(case, library)
    def build():
        c = make()
        return dataclasses.replace(c,source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
    return build
registry.candidate_factory = factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_math_child_v3.py'),run_name='__main__')
