"""Run the original arithmetic regressions and compare the same data on CPU.

All imports and calls must be inside one foreground host npu-guard command.
This diagnoses failures; it does not redefine them as original-suite passes.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
from benchmarks.ascend.guard import require_reservation
require_reservation()
import torch
import torch_npu
from workloads.device import set_run_device

torch.npu.set_device(0)
p = torch.npu.get_device_properties(0)
assert torch.npu.get_device_name(0) == 'Ascend950DT_9582'
assert (p.cube_core_num, p.vector_core_num) == (32, 64)
if torch.npu.get_device_limit(0) != {'cube_core_num': 32, 'vector_core_num': 64}:
    torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
assert torch.npu.get_device_limit(0) == {'cube_core_num': 32, 'vector_core_num': 64}
set_run_device('npu')
tests = importlib.import_module('tests.ops.test_binary_arith')
from tests.ops._div_reference import div_reference
from tileops.ops.elementwise import (DivFwdOp, FloorDivideFwdOp,
                                     LerpTensorFwdOp, RemainderFwdOp)
from workloads.elementwise import LerpCancellationWorkload


def capture(function, *args):
    frames = []
    def observe(frame, event, arg):
        if frame.f_code is function.__code__ and event in ('exception', 'return'):
            frames.append(dict(frame.f_locals))
        return observe
    error = None
    sys.settrace(observe)
    try:
        function(*args)
    except AssertionError as exc:
        error = str(exc)
    finally:
        sys.settrace(None)
    frame = next(f for f in reversed(frames) if 'a' in f and 'b' in f)
    torch.npu.synchronize()
    return frame['a'], frame['b'], error


def bitwise(actual, expected):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    nan_equal = torch.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    integer = torch.int16 if actual.element_size() == 2 else torch.int32
    mismatches = int((actual[valid].view(integer) != expected[valid].view(integer)).sum())
    different = ((actual.view(integer) != expected.view(integer)) & ~expected.isnan()).flatten()
    indices = different.nonzero().flatten()[:8].tolist()
    return dict(passed=nan_equal and mismatches == 0,
                nan_positions_equal=nan_equal, finite_bit_mismatches=mismatches,
                max_abs=float((actual.float()-expected.float()).abs().nan_to_num().max()),
                examples=[dict(index=i, actual=float(actual.flatten()[i]),
                               expected=float(expected.flatten()[i]),
                               actual_bits=int(actual.view(integer).flatten()[i]),
                               expected_bits=int(expected.view(integer).flatten()[i])) for i in indices])


def save(name, **tensors):
    path = Path(sys.argv[1]).parent / name
    torch.save({key:value.detach().cpu() for key,value in tensors.items()}, path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        cpu = value.detach().cpu().contiguous()
        h.update(str((tuple(cpu.shape), cpu.dtype)).encode())
        h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def remainder_reference(a, b):
    a, b = torch.broadcast_tensors(a.float(), b.float())
    values = []
    for x, y in zip(a.flatten().tolist(), b.flatten().tolist(), strict=True):
        if math.isinf(x) or y == 0:
            z = float('nan')
        else:
            z = math.fmod(x, y)
            if z != 0 and (y < 0) != (z < 0):
                z += y
        values.append(z)
    return torch.tensor(values, dtype=torch.float32).reshape(a.shape)


results = []
try:
    for dtype in (torch.float16, torch.bfloat16):
        workload = LerpCancellationWorkload(dtype)
        a, b, w = workload.gen_inputs()
        got = LerpTensorFwdOp()(a, b, w)
        original = workload.ref_program(a, b, w)
        cpu = torch.lerp(a.cpu().float(), b.cpu().float(), w.cpu().float()).to(dtype)
        results.append(dict(op='LerpTensorFwdOp', case=str(dtype), input_sha256=digest(a,b,w),
                tensors=save('lerp-'+str(dtype).split('.')[-1]+'.pt', a=a,b=b,w=w,
                             actual=got,cpu=cpu,original=original),
                cpu=bitwise(got, cpu), original_npu_reference=bitwise(got, original),
                expected_first=float(cpu.flatten()[0]), actual_first=float(got.cpu().flatten()[0]),
                original_first=float(original.cpu().flatten()[0])))
    a, b, error = capture(tests.test_bfloat16_div_channel_broadcast_matches_torch_bit_for_bit)
    got = DivFwdOp()(a,b)
    cpu = div_reference(a.cpu(),b.cpu())
    results.append(dict(op='DivFwdOp', case='bf16-channel-raw-bits',
            tensors=save('div-bf16-channel.pt',a=a,b=b,actual=got,cpu=cpu,original=torch.div(a,b)),
            input_sha256=digest(a,b), original_test_error=error,
            cpu=bitwise(got,cpu), original_npu_reference=bitwise(got,torch.div(a,b)),
            cpu_native_vs_independent=bitwise(torch.div(a.cpu(),b.cpu()),cpu)))
    for case_index, shapes in enumerate((((1<<16,),(1<<16,)), ((2,64,56,56),(64,1,1)))):
        a,b,error=capture(tests.test_float32_floor_ops_match_torch_bit_for_bit,*shapes)
        cpu_a,cpu_b=a.cpu(),b.cpu()
        floor=div_reference(cpu_a,cpu_b,'floor')
        remainder=remainder_reference(cpu_a,cpu_b)
        candidates=((FloorDivideFwdOp(), torch.floor_divide, floor),
                    (DivFwdOp(rounding_mode='floor'),lambda x,y:torch.div(x,y,rounding_mode='floor'),floor),
                    (RemainderFwdOp(),torch.remainder,remainder))
        for op_index,(op,reference,expected) in enumerate(candidates):
            got=op(a,b)
            results.append(dict(op=type(op).__name__, case=str(shapes),
                    tensors=save(f'floor-{case_index}-{op_index}.pt',a=a,b=b,
                                 actual=got,cpu=expected,original=reference(a,b)),
                    input_sha256=digest(a,b), original_test_error=error,
                    cpu=bitwise(got,expected), original_npu_reference=bitwise(got,reference(a,b))))
finally:
    torch.npu.synchronize()
source=Path(tests.__file__)
report=dict(status='diagnostic_completed', cases=results,
            all_cpu_bitwise=all(r['cpu']['passed'] for r in results),
            original_reference_not_relabelled=True,
            test_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            runtime_limit=torch.npu.get_device_limit(0))
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(cases=len(results),all_cpu_bitwise=report['all_cpu_bitwise'],
                     cpu_failures=[dict(op=r['op'],case=r['case'],check=r['cpu'])
                                   for r in results if not r['cpu']['passed']]),indent=2))
