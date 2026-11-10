"""Original ops-math API timing, fully inside the existing foreground guard.

The unchanged unified profiler observes the original native callable directly.
CPU transport is setup for process isolation, never a numerical fallback.
"""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src'), str(ROOT)]
p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
from benchmarks.ascend_all_pairs.ops_math.isolated import require_guard
require_guard()
import torch
import torch_npu
torch.npu.set_device(0)
from tools.measure_baselines import _time_reference, _environment
from tools.measure_baseline_cache_pair import install_exported_flush, native_fingerprints
from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
from benchmarks.ascend_all_pairs.ops_math.composition_specs import COMPOSITIONS
from benchmarks.ascend_all_pairs.ops_math.composition import make
from benchmarks.ascend_baselines import source_hashes
for op in COMPOSITIONS:
    NATIVE_CANDIDATES.setdefault(op, partial(make, op=op))

from tools.baseline_support_math_logical_v8 import install
install()

def tensors(v, fn):
    if isinstance(v, torch.Tensor): return fn(v)
    if isinstance(v, tuple): return tuple(tensors(x, fn) for x in v)
    if isinstance(v, list): return [tensors(x, fn) for x in v]
    if isinstance(v, dict): return {k: tensors(x, fn) for k, x in v.items()}
    return v

payload = torch.load(a.input, map_location='cpu', weights_only=False)
inputs = tensors(payload['inputs'], lambda t: t.to('npu'))
case = SimpleNamespace(id=payload['case_id'], params=payload['params'], inputs=inputs)
candidate = NATIVE_CANDIDATES[payload['op']](case)
impl = candidate.implementation
inputs = inputs if impl.args is None else impl.args
wall = []
child_calls = []
active = [False]
def audit(event, args):
    if active[0] and event in ('subprocess.Popen', 'os.fork', 'os.posix_spawn'):
        child_calls.append(event)
sys.addaudithook(audit)
def measured(*args):
    before = len(child_calls)
    start = time.perf_counter()
    active[0] = True
    try:
        out = impl.run(*args)
        torch.npu.synchronize()
    finally:
        active[0] = False
    wall.append({'ms': (time.perf_counter()-start)*1000, 'child': len(child_calls) != before})
    return out
try:
    flush = install_exported_flush(torch)
    timing = _time_reference(torch, torch_npu, measured, inputs, warmup=3, repeats=15, reset=impl.reset)
    if len(wall) != 19 or any(x['child'] for x in wall[-15:]):
        raise RuntimeError('Native profiler did not cover all19 direct original calls')
    torch.npu.synchronize()
    env = _environment(torch, torch_npu)
    libs = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        name = line.split()[-1]
        if name.startswith('/') and '.so' in name and any(s in name for s in ('Ascend','ascend','torch_npu','tilelang','tvm','libcust')):
            libs.add(name)
    env['loaded_native_sha256'] = native_fingerprints(libs, ROOT / 'reports/baseline-cache-20261027')
    result = {'status':'profiled_original_native', 'pid':os.getpid(), 'process_group':os.getpgrp(),
              'key':['ops-math',payload['op'],payload['case_id']],
              'timing':timing, 'wall':wall, 'timing_mode':candidate.timing_mode,
              'api':candidate.api, 'source_sha256':{**candidate.source_sha256, **source_hashes(Path(__file__))},
              'exported_flush':flush, 'environment':env, 'profiled_calls':19,
              'coverage':'All native candidate calls run directly in this profiler process; zero nested processes'}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
finally:
    if candidate.cleanup: candidate.cleanup()
    torch.npu.synchronize()
