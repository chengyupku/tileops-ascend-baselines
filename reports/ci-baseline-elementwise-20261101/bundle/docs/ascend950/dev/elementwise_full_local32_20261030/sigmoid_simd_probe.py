"""Evaluate SIMD sigmoid arithmetic against the unchanged exact low-precision contract."""
import json
from pathlib import Path
import torch
import torch_npu
import tilelang
import tilelang.ascend.language as T

def build(dtype, precision):
    @T.prim_func
    def main(X: T.Tensor((65536,), dtype), Y: T.Tensor((65536,), dtype)):
        with T.Kernel(64) as bx:
            x = T.alloc_shared((1024,), dtype)
            y = T.alloc_shared((1024,), dtype)
            T.copy(X[bx*1024:(bx+1)*1024], x)
            with T.SimdVF():
                mask = T.simd.pset(32)
                one = T.simd.vdup(1.0, 'float32')
                for i in range(16):
                    value = (T.simd.vld(x[i*64]) if dtype == 'float32' else
                             T.simd.vcvt(T.simd.vld(x[i*64], dist='UNPK_B16'), 'float32', part=0))
                    negative = T.simd.vmuls(value, -1.0, mask)
                    denominator = T.simd.vadds(T.simd.vexp(negative, mask), 1.0, mask)
                    result = T.simd.vdiv(one, denominator, mask, precision=precision)
                    if dtype == 'float32':
                        T.simd.vsts(y[i*64], result, mask)
                    else:
                        T.simd.vsts(y[i*64], T.simd.vcvt(result, dtype), mask, dist='PK_B32')
            T.copy(y, Y[bx*1024:(bx+1)*1024])
    return tilelang.compile(main, target='ascend', out_idx=-1)

rows = []
for dtype in (torch.float16, torch.bfloat16):
    x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    a = x.npu()
    expected = torch.sigmoid(a).cpu()
    for precision in ('ftz_true', 'exact'):
        kernel = build(str(dtype).split('.')[-1], precision)
        got = kernel(a).cpu()
        mismatch = got.view(torch.int16) != expected.view(torch.int16)
        indices = torch.where(mismatch)[0]
        row = {'dtype': str(dtype), 'precision': precision, 'mismatches': len(indices),
               'examples': [{'code': int(i), 'x': str(x[i].item()), 'got': str(got[i].item()),
                             'expected': str(expected[i].item()), 'got_bits': int(got.view(torch.int16)[i]),
                             'expected_bits': int(expected.view(torch.int16)[i])} for i in indices[:32]]}
        rows.append(row)
        print(row, flush=True)
        Path(__file__).with_name('sigmoid-simd-probe.json').write_text(json.dumps({'rows': rows, 'runtime_limit': torch.npu.get_device_limit(0)}, indent=2)+'\n')
torch.npu.synchronize()
