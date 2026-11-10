"""Exhaustive LUT validation followed by real AutoTuner configuration comparison."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
import torch
import torch_npu
from candidate import log_bf16_lut as K
from autotune_compat import run_search
from benchmarks.api import cases
from tools.baseline_protocol import seed_case
from workloads.device import set_run_device

torch.npu.Event = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
set_run_device('npu')
lookup = K.table()
values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
x = torch.cat([values, values[:1]])
actual = K.build(x.numel())(x.to('npu'), lookup).cpu()
reference = torch.log(x.double()).to(torch.bfloat16)
torch.testing.assert_close(actual, reference, rtol=0, atol=0, equal_nan=True)
zeros = reference == 0
assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(reference[zeros]))
probe = {'all_65536_patterns': True, 'extra_tail': True, 'exact_CPU_values': True,
         'source_sha256': hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest(),
         'table_sha256': hashlib.sha256(lookup.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
(HERE / 'edges/log-lut-exhaustive.json').write_text(json.dumps(probe, indent=2))
print('EXHAUSTIVE_LUT_PASSED', json.dumps(probe), flush=True)
case = next(c for c in cases('LogFwdOp') if c.id == '16M-bfloat16')
seed_case(torch, 'LogFwdOp', case.id)
inputs = [case.inputs[0].reshape(-1), lookup]
reference = case.reference(inputs[0].cpu()).to('npu')
n = inputs[0].numel()


def compile_fn(tile, blocks, stages):
    return K.build(n, tile, blocks, stages)


def factory(tile, blocks, stages):
    return compile_fn(tile, blocks, stages).prim_func


configs = [dict(tile=8192, blocks=64, stages=3), dict(tile=8192, blocks=64, stages=4),
           dict(tile=16384, blocks=64, stages=2)]
run_search(factory=factory, configs=configs, compile_fn=compile_fn,
           supply_prog=lambda params: inputs, reference_prog=lambda *args: reference,
           output_dir=str(HERE / 'search/log-bf16-lut'),
           source_paths=[Path(__file__), Path(K.__file__)], control_config=configs[0],
           warmup=3, repeats=15, search_backend='profile_short', profile_warmup=3,
           profile_repeats=15, rtol=.016, atol=.016)
torch.npu.synchronize()
