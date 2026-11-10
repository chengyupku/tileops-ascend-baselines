"""Scalar interpolation/extrapolation weights with cancellation and IEEE inputs."""
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from tileops.kernels.ascend.elementwise import lerp as module
from tileops.ops.elementwise import LerpScalarFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
from lerp_scalar_reference import lerp_scalar_reference

rows = []
torch.manual_seed(1701)
weights = [-2., -.5, 0., .3, .49999999, .5, .7, 1., 2.]
for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    info = torch.finfo(dtype)
    edges = torch.tensor([0., -0., 1., -1., info.tiny, -info.tiny,
                          info.max, -info.max, float('inf'), -float('inf'),
                          float('nan'), info.tiny * info.eps], dtype=dtype)
    x = torch.randn(4097).to(dtype)
    b = -x.clone()
    count = edges.numel() ** 2
    x[:count] = edges.repeat_interleave(edges.numel())
    b[:count] = edges.repeat(edges.numel())
    storage = torch.int16 if dtype.itemsize == 2 else torch.int32
    a, c = x.npu(), b.npu()
    for weight in weights:
        actual = LerpScalarFwdOp(weight=weight)(a, c).cpu()
        expected = lerp_scalar_reference(x, b, weight)
        row = dict(dtype=str(dtype), weight=weight, values=x.numel(), passed=True)
        try:
            contract = ElementwiseWorkload('LerpScalarFwdOp', (x, b), weight=weight).verification(x, b)
            compare_outputs(actual, expected, contract)
            zero = (actual == 0) & (expected == 0)
            assert torch.equal(actual.signbit()[zero], expected.signbit()[zero]), 'signed zero'
        except AssertionError as exc:
            row.update(passed=False, error=str(exc))
            bad = ~((actual == expected) | (actual.isnan() & expected.isnan()))
            row['examples'] = [dict(index=i, a=x[i].item(), b=b[i].item(),
                                    actual=actual[i].item(), expected=expected[i].item())
                               for i in bad.nonzero().flatten()[:12].tolist()]
        assert torch.equal(a.cpu().view(storage), x.view(storage))
        assert torch.equal(c.cpu().view(storage), b.view(storage))
        row['input_unchanged'] = True
        rows.append(row)
        print(row, flush=True)
torch.npu.synchronize()
base = Path(module.__file__).with_name('arithmetic.py')
Path(__file__).with_name('lerp_scalar-weights.json').write_text(json.dumps(dict(
    checks=rows, runtime_limit=torch.npu.get_device_limit(0),
    module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    reference_sha256=hashlib.sha256(Path(__file__).with_name('lerp_scalar_reference.py').read_bytes()).hexdigest(),
    dependency_manifest={str(base): hashlib.sha256(base.read_bytes()).hexdigest()}), indent=2) + '\n')
assert all(row['passed'] for row in rows)
