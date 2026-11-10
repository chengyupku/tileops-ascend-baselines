"""Decimal paths: empty/scalar/strided/tail input and explicit output canaries."""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
import tilelang
from tileops.kernels.ascend.elementwise import round as module
from tileops.ops.elementwise import RoundFwdOp
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs

rows = []
for decimals in (-1, 2):
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        op = RoundFwdOp(decimals=decimals)
        for shape in ((0, 3), (), (3, 5), (16387,)):
            x = (torch.randn(shape)*10).to(dtype)
            if len(shape) == 2 and x.numel():
                x = x.t().contiguous().t()
            a = x.npu()
            got = op(a)
            assert got.shape == x.shape and got.dtype == dtype
            assert torch.equal(a.cpu().reshape(-1).view(torch.uint8), x.reshape(-1).view(torch.uint8))
            if x.numel():
                ref = torch.round(x.float(), decimals=decimals).to(dtype)
                compare_outputs(got.cpu(), ref, ElementwiseWorkload('RoundFwdOp', (a,), decimals=decimals).verification(a))
                assert got.untyped_storage().data_ptr() != a.untyped_storage().data_ptr()
            rows.append({'dtype': str(dtype), 'decimals': decimals, 'shape': shape, 'passed': True})
            print(rows[-1], flush=True)
        n = 16387
        kernel = module.AscendRoundKernel(n, dtype, decimals)
        explicit = tilelang.compile(kernel.compiled.prim_func, target='ascend', out_idx=[])
        x = (torch.randn(n)*10).to(dtype)
        a = x.npu()
        guarded = torch.full((n+128,), 321, device='npu', dtype=dtype)
        output = guarded[64:64+n]
        output.fill_(float('nan'))
        explicit(a, output)
        torch.npu.synchronize()
        assert bool((guarded[:64] == 321).all()) and bool((guarded[64+n:] == 321).all())
        ref = torch.round(x.float(), decimals=decimals).to(dtype)
        contract = ElementwiseWorkload('RoundFwdOp', (a,), decimals=decimals).verification(a)
        compare_outputs(output.cpu(), ref, contract)
        bad = output.cpu()
        bad[-1] = float('nan')
        try:
            compare_outputs(bad, ref, contract)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted tail accepted')
        rows.append({'dtype': str(dtype), 'decimals': decimals, 'check': 'sentinel_and_negative_control', 'passed': True})
        print(rows[-1], flush=True)
torch.npu.synchronize()
Path(__file__).with_name('round-decimal-boundaries.json').write_text(json.dumps({'rows': rows, 'runtime_limit': torch.npu.get_device_limit(0), 'module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}, indent=2)+'\n')
