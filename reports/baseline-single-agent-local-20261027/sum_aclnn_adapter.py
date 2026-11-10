"""Original complete aclnnReduceSum with unchanged input layout and precision."""
import ctypes as C
import json
from pathlib import Path


def factory(case, output, pins):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from benchmarks.ascend_math_more import _MathCall
    assert case.id == "mlp-intermediate-float32"
    assert case.params["dim"] == -1 and not case.params["keepdim"]
    binding = _MathCall("aclnnReduceSum", [C.c_void_p, C.c_void_p, C.c_bool,
                                         C.c_int, C.c_void_p])

    def run(x):
        assert tuple(x.shape) == (2048, 11008) and x.dtype == torch.float32
        out = torch.empty((2048,), dtype=x.dtype, device=x.device)
        record = binding.record()
        args = [binding.tensor(x.contiguous(), record), binding.array([-1], record),
                False, binding.codes[x.dtype], binding.tensor(out, record)]
        print("ORIGINAL_REDUCESUM_ACLNN_ENTERED", flush=True)
        binding.call(args, record, x.device)
        print("ORIGINAL_REDUCESUM_ACLNN_HOST_RETURNED", flush=True)
        torch.npu.synchronize()
        (output / "native.json").write_text(json.dumps({"api": binding.api,
            "binary": str(binding.binary), "synchronized": True,
            "input": list(x.shape), "output": list(out.shape)}, indent=2))
        (output / "native_maps.txt").write_text(Path("/proc/self/maps").read_text())
        return out

    return Candidate(Implementation(run), "ops-math::aclnnReduceSum",
                     "Original full public ACLNN reduction, axis -1, FP32; "
                     "input and output layout unchanged, no split or external sum.",
                     pins, cleanup=binding.cleanup)
