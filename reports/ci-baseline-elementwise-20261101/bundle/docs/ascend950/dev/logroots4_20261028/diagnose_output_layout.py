"""Hold arithmetic/input fixed and vary only output address; guarded diagnostic."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
import torch
import torch_npu
import tilelang
from benchmarks.api import cases
from tools.baseline_protocol import seed_case
from tools.measure_baselines import _time_reference
from full_digest import tensor_digest
from tileops.kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops.ops.elementwise import Log1pFwdOp
from workloads.device import set_run_device

torch.npu.Event = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
set_run_device('npu')
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

x = xs[0].reshape(-1)
n = x.numel()
kernel = AscendLog1pKernel(n, x.dtype).compiled
explicit = tilelang.compile(kernel.prim_func, target='ascend', out_idx=[])
assert explicit.get_kernel_source() == kernel.get_kernel_source()
base = torch.empty(n + 4 * 1024 * 1024 + 128, device='npu', dtype=x.dtype)
rows = []
for offset in (0, 64, 2048, 524288, 4194304):
    output = base[offset:offset + n]

    def fn(value):
        explicit(value, output)
        return output

    actual = fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().reshape(ref.shape), ref, rtol=1e-5, atol=1e-5)
    profile = _time_reference(torch, torch_npu, fn, (x,), 3, 15)
    row = dict(offset_bytes=offset * 4, input_pointer=x.data_ptr(), output_pointer=output.data_ptr(),
               profile=profile, input_digest=before, correct=True,
               generated_c_sha256=hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest(),
               purpose='diagnostic_only_not_acceptance')
    rows.append(row)
    (HERE / 'search/output-layout-v3.json').write_text(json.dumps(rows, indent=2))
    print('OUTPUT_LAYOUT', offset * 4, profile['latency_ms'] * 1000, flush=True)
assert tensor_digest(xs) == before
torch.npu.synchronize()
