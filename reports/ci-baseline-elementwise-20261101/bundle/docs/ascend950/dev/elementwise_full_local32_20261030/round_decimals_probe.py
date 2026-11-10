"""Public dispatch decimal scaling across full low-precision input domains."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import round as module
from tileops.ops.elementwise import RoundFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs

torch.manual_seed(20261030)
rows = []
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    if dtype != torch.float32:
        base = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    else:
        base = torch.randint(0, 2**32, (65536,), dtype=torch.int64).to(torch.int32).view(dtype)
    x = torch.cat((base, torch.tensor([0., -0., .05, -.05, .15, -.15, 100., -100., 1e-38, -1e-38], dtype=dtype)))
    a = x.npu()
    for decimals in (-39, -4, -1, 1, 2, 4, 38, 39):
        op = RoundFwdOp(decimals=decimals)
        got = op(a).cpu()
        expected = torch.round(x.float(), decimals=decimals).to(dtype)
        compare_outputs(got, expected, ElementwiseWorkload('RoundFwdOp', (a,), decimals=decimals).verification(a))
        valid = ~torch.isnan(expected)
        assert torch.equal(torch.signbit(got[valid]), torch.signbit(expected[valid]))
        assert torch.equal(a.cpu().view(torch.uint8), x.view(torch.uint8))
        changed = torch.full_like(a, 123.25)
        second = op(changed).cpu()
        compare_outputs(second, torch.round(changed.cpu().float(), decimals=decimals).to(dtype), ElementwiseWorkload('RoundFwdOp', (changed,), decimals=decimals).verification(changed))
        rows.append({'dtype': str(dtype), 'decimals': decimals, 'values': len(x), 'passed': True})
        print(rows[-1], flush=True)
        Path(__file__).with_name('round-decimals-progress.json').write_text(json.dumps(rows))
torch.npu.synchronize()
Path(__file__).with_name('round-decimals-probe.json').write_text(json.dumps({'rows': rows, 'runtime_limit': torch.npu.get_device_limit(0), 'module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}, indent=2)+'\n')
