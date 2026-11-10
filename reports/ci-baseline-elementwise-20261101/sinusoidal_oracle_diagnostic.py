"""Compare unchanged production output with both original NPU and CPU references.

Only execute this complete foreground command under the host npu-guard.
The original suite, and its failure, remain separate evidence.
"""
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path
import sys

BASE = Path('/opt/tileops-ci/server2-20261030/overnight-20261101')
ROOT = BASE / 'ci-full-v3/workspace'
sys.path[:0] = [str(ROOT), str(ROOT / 'TileOPs'), str(ROOT / 'TileOPs/src')]
from benchmarks.ascend.guard import require_reservation
require_reservation()
import torch
import torch_npu
from tileops.ops.elementwise import SinusoidalFwdOp
from workloads.device import set_run_device
from workloads.elementwise import ElementwiseWorkload, sinusoidal_reference
from workloads.numerics import compare_outputs, reference_tolerance

torch.npu.set_device(0)
p = torch.npu.get_device_properties(0)
assert torch.npu.get_device_name(0) == 'Ascend950DT_9582'
assert (p.cube_core_num, p.vector_core_num) == (32, 64)
if torch.npu.get_device_limit(0) != {'cube_core_num': 32, 'vector_core_num': 64}:
    torch.npu.set_device_limit(0, cube_num=32, vector_num=64)
assert torch.npu.get_device_limit(0) == {'cube_core_num': 32, 'vector_core_num': 64}
set_run_device('npu')
results = []
try:
    for seq_len, d_model, dtype in ((512, 256, torch.float16),
                                  (512, 256, torch.float32),
                                  (65, 130, torch.float16)):
        op = SinusoidalFwdOp(seq_len=seq_len, d_model=d_model, out_dtype=dtype, device='npu')
        output = op()
        cpu_reference = sinusoidal_reference(seq_len, d_model, dtype, device='cpu')
        original_reference = sinusoidal_reference(seq_len, d_model, dtype, device='npu')
        torch.npu.synchronize()
        actual = output.cpu()
        original = original_reference.cpu()
        evidence = ElementwiseWorkload('SinusoidalFwdOp', (), seq_len=seq_len,
                    d_model=d_model, out_dtype=dtype, device='npu').verification()
        check = compare_outputs(actual, cpu_reference, evidence)
        assert check.checked_outputs == check.total_outputs == 1
        original_error = None
        try:
            compare_outputs(actual, original, evidence)
        except AssertionError as exc:
            original_error = str(exc)
        built = [k for entries in op._built_entries.values() for k in entries.values()]
        assert built and all(type(k).__name__ == 'AscendSinusoidalKernel' for k in built)
        kernel_module = importlib.import_module(type(built[0]).__module__)
        kernel_file = Path(kernel_module.__file__)
        source_sha = hashlib.sha256(kernel_file.read_bytes()).hexdigest()
        expected = json.loads((ROOT / 'ELEMENTWISE_EXPORT.json').read_text())['source_files'][
            'src/tileops/kernels/ascend/elementwise/sinusoidal.py']
        assert source_sha == expected == '1326d01b7f55d25d0c9d323b6bd63f6f2850b19eb2c92538e0f2cb118bcb12d6'
        results.append(dict(seq_len=seq_len, d_model=d_model, dtype=str(dtype),
                cpu_check=dataclasses.asdict(check), unchanged_evidence=repr(evidence),
                dtype_default_tolerance=reference_tolerance(dtype),
                original_npu_reference_passed=original_error is None,
                original_npu_reference_error=original_error,
                kernel_cpu_max_abs=float((actual.float()-cpu_reference.float()).abs().max()),
                kernel_npu_reference_max_abs=float((actual.float()-original.float()).abs().max()),
                cpu_npu_reference_max_abs=float((cpu_reference.float()-original.float()).abs().max()),
                kernel_source_sha256=source_sha,
                runtime_limit=torch.npu.get_device_limit(0)))
finally:
    torch.npu.synchronize()
out = Path(sys.argv[1])
out.write_text(json.dumps(dict(status='passed', original_reference_not_relabelled=True,
                   production_kernel_unchanged=True, cases=results), indent=2)+'\n')
print(json.dumps(dict(status='passed', cases=len(results),
                     original_reference_failed=sum(not r['original_npu_reference_passed'] for r in results))))
