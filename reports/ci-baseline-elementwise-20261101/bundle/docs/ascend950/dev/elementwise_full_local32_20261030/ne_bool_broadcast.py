"""All Boolean byte pairs across same, bias and channel layouts, both orders."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from tileops.kernels.ascend.elementwise import comparison as module
from tileops.ops.elementwise import NeFwdOp

values = torch.arange(256, dtype=torch.uint8)
rows = []
for layout in ['same', 'bias', 'channel']:
    if layout == 'same':
        x, b = values.repeat_interleave(256), values.repeat(256)
    elif layout == 'bias':
        x, b = values[:, None].repeat(1, 256), values[None, :]
    else:
        x, b = values[:, None, None].repeat(1, 256, 64), values[:, None]
    if x.ndim >= 2:
        x = x.transpose(0, 1).contiguous().transpose(0, 1)
    for swapped in [False, True]:
        left, right = (b, x) if swapped else (x, b)
        a, c = left.npu().view(torch.bool), right.npu().view(torch.bool)
        actual = NeFwdOp()(a, c).cpu()
        expected = (left != 0) != (right != 0)
        assert actual.shape == expected.shape and actual.dtype == torch.bool
        assert torch.equal(actual, expected)
        assert bool((actual.view(torch.uint8) <= 1).all()), 'noncanonical bool output'
        assert torch.equal(a.view(torch.uint8).cpu(), left)
        assert torch.equal(c.view(torch.uint8).cpu(), right)
        row = dict(layout=layout, swapped=swapped, pairs=65536,
                   output_elements=actual.numel(), passed=True,
                   canonical_output=True, input_unchanged=True)
        rows.append(row)
        print(row, flush=True)
torch.npu.synchronize()
Path(__file__).with_name('ne-bool-broadcast.json').write_text(json.dumps(dict(
    checks=rows, module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    runtime_limit=torch.npu.get_device_limit(0),
    dependency_manifest={str(Path(module.__file__).with_name(name)):
                         hashlib.sha256(Path(module.__file__).with_name(name).read_bytes()).hexdigest()
                         for name in []}
), indent=2) + '\n')
