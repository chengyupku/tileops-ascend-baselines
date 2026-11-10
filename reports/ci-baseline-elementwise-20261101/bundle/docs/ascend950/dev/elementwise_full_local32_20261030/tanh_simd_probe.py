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
                    magnitude = T.simd.vabs(value, mask)
                    exponent = T.simd.vexp(T.simd.vmuls(magnitude, -2.0, mask), mask)
                    numerator = T.simd.vsub(one, exponent, mask)
                    denominator = T.simd.vadds(exponent, 1.0, mask)
                    raw = T.simd.vdiv(numerator, denominator, mask, precision="ftz_true")
                    square = T.simd.vmul(magnitude, magnitude, mask)
                    correction = T.simd.vadds(T.simd.vmuls(square, -1.0 / 3.0, mask), 1.0, mask)
                    polynomial = T.simd.vmul(magnitude, correction, mask)
                    small = T.simd.vcmps(magnitude, 0.015625, mask, op="le")
                    result = T.simd.vsel(polynomial, raw, small)
                    tiny = T.simd.vcmps(magnitude, 0.000244140625, mask, op="le")
                    result = T.simd.vsel(magnitude, result, tiny)
                    sign = T.simd.vand(T.reinterpret(value, "int32x64"), T.simd.vdup(T.int32(-2147483648), "int32"), mask)
                    result = T.reinterpret(T.simd.vor(T.reinterpret(result, "int32x64"), sign, mask), "float32x64")
                    if dtype == 'float32':
                        T.simd.vsts(y[i*64], result, mask)
                    else:
                        T.simd.vsts(y[i*64], T.simd.vcvt(result, dtype), mask, dist='PK_B32')
            T.copy(y, Y[bx*1024:(bx+1)*1024])
    return tilelang.compile(main, target='ascend', out_idx=-1)

from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs
rows = []
torch.manual_seed(20261030)
for dtype in (torch.float16, torch.bfloat16, torch.float32):
    if dtype != torch.float32:
        x = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    else:
        x = torch.cat([torch.randint(0,2**32,(32768,),dtype=torch.int64).to(torch.int32).view(dtype),torch.linspace(-8,8,32768)])
        x[:10] = torch.tensor([0.,-0.,1e-45,-1e-45,1e-38,-1e-38,1/64,-1/64,float('inf'),float('nan')])
    a=x.npu();kernel=build(str(dtype).split('.')[-1], 'ftz_true');got=kernel(a).cpu();expected=torch.tanh(x.double()).to(dtype)
    error=None
    try:
        compare_outputs(got,expected,ElementwiseWorkload('TanhFwdOp',(a,)).verification(a))
        zeros=expected==0
        assert torch.equal(torch.signbit(got[zeros]),torch.signbit(expected[zeros]))
    except AssertionError as e:error=str(e)
    diff=(got.float()-expected.float()).abs();indices=torch.argsort(diff.nan_to_num(nan=-1),descending=True)[:8]
    rows.append({'dtype':str(dtype),'passed':error is None,'error':error,'worst':[{'x':str(x[i].item()),'got':str(got[i].item()),'cpu_fp64':str(expected[i].item())} for i in indices]})
    print(rows[-1],flush=True)
    Path(__file__).with_name('tanh-simd-probe.json').write_text(json.dumps({'rows':rows,'runtime_limit':torch.npu.get_device_limit(0)},indent=2)+'\n')
torch.npu.synchronize()
assert all(v['passed'] for v in rows)
