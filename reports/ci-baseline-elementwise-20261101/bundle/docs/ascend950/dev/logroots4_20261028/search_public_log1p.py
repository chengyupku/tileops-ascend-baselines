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
from tileops.kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops.ops.elementwise import Log1pFwdOp
from workloads.device import set_run_device

torch.npu.Event = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
set_run_device('npu')
out = HERE / 'search/public-configs-log1p-f32-v1'
out.mkdir(exist_ok=False)
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
    Path(__file__), Path(K.__file__), Path(K.__file__).with_name('log1p.py'),
    ROOT / 'tools/measure_baselines.py')}
(out / 'sources.json').write_text(json.dumps(sources, indent=2))

for case in cases('Log1pFwdOp'):
    seed_case(torch, 'Log1pFwdOp', case.id)
    xs = case.inputs
    before = tensor_digest(xs)
    ref = case.reference(*(x.cpu() for x in xs))
    op = Log1pFwdOp(**case.arguments)
    got = op(*xs)
    torch.npu.synchronize()
    if case.id == '16M-float32':
        break
    _time_reference(torch, torch_npu, op, tuple(xs), 3, 15)
    assert tensor_digest(xs) == before

n = xs[0].numel()
specs = [(8192, 64, 3), (4096, 64, 3), (4096, 64, 4), (4096, 64, 6),
         (8192, 128, 3), (4096, 128, 4), (2048, 64, 4), (8192, 32, 3)]
configs = [dict(tile=t, blocks=b, stages=s) for t, b, s in specs]
identities = {}
rows = []


def compile_candidate(**config):
    torch.npu.set_device(0)
    config.pop('__pass_configs__', None)
    kernel = K.build(n, 'float32', 'log1p', **config)
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
        old_configs = AscendLog1pKernel.configs
        AscendLog1pKernel.configs = {
            **old_configs, (n, 'float32'): tuple(config[k] for k in ('tile', 'blocks', 'stages'))}
        row = {'config': config, 'status': 'failed', 'purpose': 'public_entry_configuration_comparison'}
        try:
            public = Log1pFwdOp()
            actual = public(*xs)
            torch.npu.synchronize()
            torch.testing.assert_close(actual.cpu(), ref, rtol=1e-5, atol=1e-5)
            built = [v for group in public._built_entries.values() for v in group.values()]
            assert len(built) == 1 and built[0].compiled is kernel
            profile = _time_reference(torch, torch_npu, public, tuple(xs), 3, 15)
            assert tensor_digest(xs) == before
            row.update(status='ok', correct=True, input_unchanged=True, input_digest=before,
                       profile=profile, input_pointer=xs[0].data_ptr(),
                       warmup=3, repeats=15, event_used=False)
            kwargs['benchmark_state'].jit_input_tensors = list(xs)
            return profile['latency_ms'], None
        except BaseException as exc:
            row['error'] = repr(exc)
            raise
        finally:
            AscendLog1pKernel.configs = old_configs
            rows.append(row)
            (out / 'trials.json').write_text(json.dumps(rows, indent=2))
            print('PUBLIC_TRIAL', json.dumps(row), flush=True)


tuner = PublicTuner.from_kernel(factory, configs=configs)
tuner.set_compile_args(target='ascend', out_idx=[-1], execution_backend='auto')
tuner.set_profile_args(supply_prog=lambda params: xs, ref_prog=lambda *args: ref,
                       rtol=1e-5, atol=1e-5, cache_input_tensors=False, backend='msprof')
tuner.jit_compile = compile_candidate
tuner.jit_elaborate = lambda **config: compile_candidate(**config).prim_func
os.chdir(out)
tuner.run(warmup=3, rep=15, timeout=0, use_pipeline=False,
          enable_grouped_compile=False, benchmark_multi_gpu=False, early_stop=False)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
assert len(rows) == len(configs), 'Every candidate must be reported'
torch.npu.synchronize()
