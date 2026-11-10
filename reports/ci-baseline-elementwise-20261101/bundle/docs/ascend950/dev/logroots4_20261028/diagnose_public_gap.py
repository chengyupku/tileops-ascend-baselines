"""Same-process input/call-path comparison; diagnostic only, under npu-guard."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
import torch
import torch_npu
from benchmarks.api import cases
from tools.baseline_protocol import seed_case
from tools.measure_baselines import _time_reference
from full_digest import tensor_digest
from tileops.kernels.ascend.elementwise.log1p import AscendLog1pKernel
from tileops.ops.elementwise import Log1pFwdOp
from workloads.device import set_run_device

torch.npu.Event = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Event forbidden'))
torch.npu.set_device(0)
set_run_device('npu')
out = HERE / 'search/public-gap-v3.json'
rows = []

# Reproduce the preceding two manifest cases and allocator history of pilot.py.
for case in cases('Log1pFwdOp'):
    seed_case(torch, 'Log1pFwdOp', case.id)
    xs = case.inputs
    before = tensor_digest(xs)
    ref = case.reference(*(x.cpu() for x in xs))
    op = Log1pFwdOp(**case.arguments)
    got = op(*xs)
    torch.npu.synchronize()
    torch.testing.assert_close(got.cpu(), ref, rtol=1e-3 if got.dtype == torch.float16 else .016 if got.dtype == torch.bfloat16 else 1e-5,
                               atol=1e-3 if got.dtype == torch.float16 else .016 if got.dtype == torch.bfloat16 else 1e-5)
    if case.id == '16M-float32':
        break
    _time_reference(torch, torch_npu, op, tuple(xs), 3, 15)
    assert tensor_digest(xs) == before

kernel = AscendLog1pKernel(xs[0].numel(), xs[0].dtype).compiled
source_hash = hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
original = xs[0]
clone = original.clone()
assert tensor_digest([clone]) == before


def measure(label, public, value):
    torch.npu.set_device(0)
    outputs = []

    def fn(x):
        result = op(x) if public else kernel(x.reshape(-1)).reshape(x.shape)
        outputs.append(result.data_ptr())
        return result

    profile = _time_reference(torch, torch_npu, fn, (value,), 3, 15)
    row = dict(label=label, input_pointer=value.data_ptr(), output_pointers=sorted(set(outputs)),
               input_stride=value.stride(), input_digest=tensor_digest([value]),
               generated_c_sha256=source_hash, profile=profile,
               purpose='diagnostic_only_not_acceptance')
    rows.append(row)
    out.write_text(json.dumps(rows, indent=2))
    print(label, profile['latency_ms'] * 1000, flush=True)


for label, public, value in (
    ('public_original', True, original),
    ('direct_original', False, original),
    ('public_clone', True, clone),
    ('direct_clone', False, clone),
):
    measure(label, public, value)
with ThreadPoolExecutor(max_workers=1) as pool:
    pool.submit(measure, 'thread_direct_original', False, original).result()
measure('public_original_repeat', True, original)
assert tensor_digest([original]) == before
torch.npu.synchronize()
