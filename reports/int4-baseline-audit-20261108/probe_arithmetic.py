"""Local diagnostic only: exact packing and synchronized host timing, no release."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
torch.npu.set_device(0)
torch.npu.set_option({'ACL_PRECISION_MODE': 'must_keep_origin_dtype'})

def pack_original(q):
    q = q.to(torch.int8)
    return (q[..., 0] << 4) | (q[..., 1] & 15)

def pack_arithmetic(q):
    low = q[..., 1]
    low_unsigned = torch.where(low < 0, low + 16, low)
    return (q[..., 0] * 16 + low_unsigned).to(torch.int8)

def quant(w, group_size, pack):
    n, k = w.shape
    groups = w.float().view(-1, group_size)
    lo, hi = groups.amin(-1), groups.amax(-1)
    multiplier = torch.where(hi == lo, 1.0, 16.0 / (hi - lo))
    offset = (hi + lo) * 0.5
    q = ((groups - offset[:, None]) * multiplier[:, None]).round().clamp(-8, 7)
    return pack(q.view(n, k // 2, 2)), torch.stack((1.0 / multiplier, offset), -1)

codes = torch.tensor([(a, b) for a in range(-8, 8) for b in range(-8, 8)], dtype=torch.float32)
expected = pack_original(codes)
actual = pack_arithmetic(codes.npu()).cpu()
assert torch.equal(actual, expected)
torch.manual_seed(1234)
checks = []
for shape, group in [((32, 64), 64), ((32, 128), 128), ((32, 4096), 4096), ((4096, 4096), 64)]:
    w = torch.randn(shape, dtype=torch.float16).npu()
    w[0].zero_()
    w[1].fill_(2)
    ref = quant(w, group, pack_original)
    got = quant(w, group, pack_arithmetic)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    checks.append({'shape': shape, 'group_size': group, 'packed_and_params_exact': True})

timings = {'original': [], 'arithmetic': []}
functions = {'original': pack_original, 'arithmetic': pack_arithmetic}
for name in ['original', 'arithmetic'] * 4:
    torch.npu.synchronize()
    start = time.perf_counter()
    result = quant(w, 64, functions[name])
    torch.npu.synchronize()
    timings[name].append((time.perf_counter() - start) * 1000)

parser_file = ROOT.parent / 'torch-all2208-20261105/tools/kernel_task_profile_full32.py'
spec = importlib.util.spec_from_file_location('raw_parser', parser_file)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
profiles = {}
for name, pack in functions.items():
    os.environ['ASCEND_WORK_PATH'] = str(ROOT / ('local_profile_' + name))
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=lambda _: None,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False, data_simplification=False),
    ) as prof:
        result = quant(w, 64, pack)
        torch.npu.synchronize()
    path = Path(prof.prof_if.prof_path)
    parsed = parser.parse_profile(path, 1)
    profiles[name] = {'path': str(path), 'trace_sha256': parser.trace_digest(path), **parsed}

assert profiles['arithmetic']['profile_aicpu_task_count'] == 0
props = torch.npu.get_device_properties(0)
report = {
    'scope': 'Local diagnostic, not production 32-core benchmark; warm synchronized host samples without flush',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'device': torch.npu.get_device_name(0), 'cube': props.cube_core_num, 'vector': props.vector_core_num,
    'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
    'exhaustive_code_pairs': 256, 'checks': checks,
    'host_samples_ms': timings, 'host_mean_ms': {k: statistics.mean(v) for k, v in timings.items()},
    'profiles': profiles,
}
(ROOT / 'ARITHMETIC_LOCAL.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'profiles'}, indent=2))
print('AICPU counts:', {k: v['profile_aicpu_task_count'] for k, v in profiles.items()})
torch.npu.synchronize()
