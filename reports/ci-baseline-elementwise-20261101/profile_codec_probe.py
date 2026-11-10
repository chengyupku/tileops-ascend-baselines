"""Retain a tiny diagnostic trace to verify device-record aggregation.

Run under npu-guard after the baseline campaign completes. These traces validate
the record interpretation only; none of their timings are admitted as benchmarks.
"""
import collections
import json
from pathlib import Path
import shutil
import struct
import sys

workspace, output = map(Path, sys.argv[1:])
sys.path[:0] = [str(workspace), str(workspace / 'TileOPs'), str(workspace / 'TileOPs/src')]
from benchmarks.ascend.guard import require_reservation
require_reservation()
import torch
import torch_npu
from tools.measure_baselines import _parse_ffts_profile, _resolve_kernel_names

output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device(0)
p = torch.npu.get_device_properties(0)
assert (p.cube_core_num, p.vector_core_num) == (32, 64)
assert torch.npu.get_device_name(0) == 'Ascend950DT_9582'
x = torch.randn((256, 256), dtype=torch.float16, device='npu')
y = torch.randn_like(x)
q = torch.randn((1, 4, 64, 64), dtype=torch.float16, device='npu')
summary = []
try:
    for name, fn in [('matmul', lambda: torch.mm(x, y)),
                     ('attention', lambda: torch.nn.functional.scaled_dot_product_attention(q, q, q))]:
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        directory = output / name
        directory.mkdir()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, data_simplification=False)) as prof:
            value = fn()
            torch.npu.synchronize()
            prof.step()
        path = Path(prof.prof_if.prof_path)
        names = _resolve_kernel_names(path)
        groups = collections.defaultdict(list)
        for file in path.rglob('ffts_profile*'):
            if file.name.endswith('.done'):
                continue
            raw = file.read_bytes()
            for offset in range(0, len(raw) // 128 * 128, 128):
                v = struct.unpack_from('<16q', raw, offset)
                if v[1] and v[15] >= v[14]:
                    seq = (v[0] >> 32) & 65535
                    groups[seq].append(dict(core='AIV' if v[2] >= 2**31 else 'AIC',
                                            start_ns=v[14], end_ns=v[15], duration_ns=v[15]-v[14]))
        by_task = [dict(seq=seq, name=names.get(seq), records=records,
                        both_core_roles=len({r['core'] for r in records}) > 1)
                   for seq, records in groups.items()]
        report = dict(operation=name, profile_path=str(path), formal_benchmark=False,
                      project_parser=_parse_ffts_profile(path, repeats=1), tasks=by_task,
                      mixed_same_task=sum(t['both_core_roles'] for t in by_task),
                      exported_csv=[str(p) for p in directory.rglob('*.csv')])
        (directory / 'codec.json').write_text(json.dumps(report, indent=2) + '\n')
        summary.append(report)
finally:
    torch.npu.synchronize()
(output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps([{k:v for k,v in x.items() if k != 'tasks'} for x in summary]))
