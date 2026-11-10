"""Actual AutoTuner, with CPU oracle and public dispatch for each candidate."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
import torch
import torch_npu
from tilelang.autotuner import AutoTuner
from benchmarks.api import cases
from tools.baseline_protocol import seed_case
from tools.measure_baselines import _time_reference
from full_digest import tensor_digest
from tileops.kernels.ascend.elementwise import _logroots as K
import argparse
import importlib
parser = argparse.ArgumentParser()
parser.add_argument('--op', required=True, choices=['LogFwdOp', 'SqrtFwdOp', 'RsqrtFwdOp'])
options = parser.parse_args()
kind = {'LogFwdOp': 'log', 'SqrtFwdOp': 'sqrt', 'RsqrtFwdOp': 'rsqrt'}[options.op]
Kernel = getattr(importlib.import_module('tileops.kernels.ascend.elementwise.' + kind), 'Ascend' + options.op.removesuffix('FwdOp') + 'Kernel')
Op = getattr(importlib.import_module('tileops.ops.elementwise'), options.op)
from workloads.device import set_run_device

torch.npu.Event = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
set_run_device('npu')
out = HERE / 'search' / ('public-configs-' + options.op + '-bf16-deeper')
out.mkdir(exist_ok=False)
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    Path(__file__), Path(K.__file__), Path(K.__file__).with_name(kind + '.py'),
    ROOT / 'tools/measure_baselines.py')}
(out / 'sources.json').write_text(json.dumps(sources, indent=2))

for case in cases(options.op):
    seed_case(torch, options.op, case.id)
    xs = case.inputs
    before = tensor_digest(xs)
    ref = case.reference(*(x.cpu() for x in xs))
    op = Op(**case.arguments)
    got = op(*xs)
    torch.npu.synchronize()
    if case.id == '16M-bfloat16':
        break
    _time_reference(torch, torch_npu, op, tuple(xs), 3, 15)
    assert tensor_digest(xs) == before

n = xs[0].numel()
specs = [(8192, 64, 3), (8192, 64, 4), (8192, 64, 6), (16384, 64, 3)]
configs = [dict(tile=t, blocks=b, stages=s) for t, b, s in specs]
identities = {}
rows = []


def compile_candidate(**config):
    torch.npu.set_device(0)
    config.pop('__pass_configs__', None)
    kernel = K.build(n, 'bfloat16', kind, config['tile'], config['blocks'], config['stages'])
    identities[id(kernel)] = config.copy()
    key = '-'.join(str(config[k]) for k in ('tile', 'blocks', 'stages'))
    (out / (key + '.c')).write_text(kernel.get_kernel_source())
    return kernel


def factory(tile, blocks, stages):
    return compile_candidate(tile=tile, blocks=blocks, stages=stages).prim_func


class PublicTuner(AutoTuner):
    def _benchmark_target(self, *args, **kwargs):
        torch.npu.set_device(0)
        kernel = kwargs['jit_kernel']
        config = identities[id(kernel)]
        old_configs = Kernel.configs
        Kernel.configs = {
            **old_configs, (n, 'bfloat16'): tuple(config[k] for k in ('tile', 'blocks', 'stages'))}
        row = {'op': options.op, 'config': config, 'status': 'failed', 'purpose': 'public_entry_configuration_comparison'}
        try:
            public = Op()
            actual = public(*xs)
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), ref, rtol=.016, atol=.016)
            built = [v for group in public._built_entries.values() for v in group.values()]
            assert len(built) == 1
            assert built[0].compiled.get_kernel_source() == kernel.get_kernel_source()
            profile = _time_reference(torch, torch_npu, public, tuple(xs), 3, 15)
            assert tensor_digest(xs) == before
            row.update(status='ok', correct=True, input_unchanged=True, input_digest=before,
                       profile=profile, input_pointer=xs[0].data_ptr(),
                       warmup=3, repeats=15, event_used=False)
            kwargs['benchmark_state'].jit_input_tensors = list(xs)
            return profile['latency_ms'], None
        except BaseException as exc:
            row['error'] = __import__('traceback').format_exc()
            raise
        finally:
            Kernel.configs = old_configs
            rows.append(row)
            (out / 'trials.json').write_text(json.dumps(rows, indent=2))
            print('PUBLIC_TRIAL', json.dumps(row), flush=True)


tuner = PublicTuner.from_kernel(factory, configs=configs)
tuner.set_compile_args(target='ascend', out_idx=[-1], execution_backend='auto')
tuner.set_profile_args(supply_prog=lambda params: xs, ref_prog=lambda *args: ref,
                       rtol=.016, atol=.016, cache_input_tensors=False, backend='msprof')
tuner.jit_compile = compile_candidate
tuner.jit_elaborate = lambda **config: compile_candidate(**config).prim_func
os.chdir(out)
tuner.run(warmup=3, rep=15, timeout=0, use_pipeline=False,
          enable_grouped_compile=False, benchmark_multi_gpu=False, early_stop=False)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
assert len(rows) == len(configs), 'Every candidate must be reported'
torch.npu.synchronize()
