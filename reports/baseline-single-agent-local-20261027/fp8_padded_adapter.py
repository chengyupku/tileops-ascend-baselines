"""One original GroupedMatmulV5 invocation, with zero-row storage padding.

Each independently scaled weight row occupies the first row of its own native
128-row scale block. The other rows are positive zero. Extracting the first
column of each output block recovers the complete requested GEMM. No K split,
dequantization, scale rounding, or arithmetic outside the original full API.
All packing, allocation, output extraction and synchronization occur in run.
"""
import ctypes as C
import json
from pathlib import Path


def factory(case, output, pins):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    from benchmarks.ascend_math_more import _MathCall

    if len(case.inputs) != 4:
        raise UnsupportedCase("This route requires bias-free FP8 GEMM.")
    a, w, sa, sw = case.inputs
    m, k = a.shape
    n = w.shape[0]
    if (a.dtype != torch.float8_e4m3fn or w.dtype != a.dtype
            or tuple(w.shape) != (n, k) or k % 128
            or tuple(sa.shape) != (m, k // 128)
            or tuple(sw.shape) != (n, k // 128)
            or sa.dtype != torch.float32 or sw.dtype != torch.float32):
        raise UnsupportedCase("Independent FP32 row/K128 scales required.")
    dtype = getattr(torch, case.params["out_dtype"])
    vendor = Path(__file__).resolve().parent / "ops-transformer-vendor"
    binding = _MathCall("aclnnGroupedMatmulV5",
                        [C.c_void_p] * 12 + [C.c_int64] * 4 + [C.c_void_p] * 4,
                        library="ops-transformer",
                        host_library=vendor / "op_api/lib/libcust_opapi.so")
    base = binding.base
    base.aclCreateTensorList.argtypes = [C.POINTER(C.c_void_p), C.c_uint64]
    base.aclCreateTensorList.restype = C.c_void_p
    base.aclDestroyTensorList.argtypes = [C.c_void_p]
    base.aclDestroyTensorList.restype = C.c_int
    lists = []
    calls = []

    def tensorlist(tensors, record):
        ptrs = [binding.tensor(t, record) for t in tensors]
        array = (C.c_void_p * len(ptrs))(*ptrs)
        record["keep"].append(array)
        ptr = base.aclCreateTensorList(array, len(ptrs))
        if not ptr:
            raise RuntimeError("aclCreateTensorList failed")
        for t in ptrs:
            record["tensors"].remove(t)
        lists.append(ptr)
        return ptr

    def cleanup():
        torch.npu.synchronize()
        while lists:
            binding.check(base.aclDestroyTensorList(lists.pop()), "aclDestroyTensorList")
        binding.cleanup()

    def run(a, w, sa, sw):
        # Swap operands when that reduces the padded weight storage.
        swap = a.shape[0] <= w.shape[0]
        lhs, weights, lhs_scales, weight_scales = (w, a, sw, sa) if swap else (a, w, sa, sw)
        rows, inner = weights.shape
        padded_bytes = torch.zeros((rows, 128, inner), dtype=torch.uint8, device=a.device)
        padded_bytes[:, 0, :].copy_(weights.contiguous().view(torch.uint8))
        rhs = padded_bytes.reshape(rows * 128, inner).T.contiguous().view(a.dtype).unsqueeze(0)
        lhs = lhs.contiguous()
        scale_a = lhs_scales.contiguous()
        scale_b = weight_scales.T.contiguous().unsqueeze(0)
        out = torch.empty((lhs.shape[0], rows * 128), dtype=dtype, device=a.device)
        groups = torch.full((1,), lhs.shape[0], dtype=torch.int64, device=a.device)
        record = binding.record()
        args = [tensorlist([lhs], record), tensorlist([rhs], record), None,
                tensorlist([scale_b], record), None, None, None,
                tensorlist([scale_a], record), binding.tensor(groups, record),
                None, None, None, 3, 0, 0, 0, None,
                tensorlist([out], record), None, None]
        evidence = {"api": binding.api, "binary": str(binding.binary),
                    "swapped": swap, "native_shapes": [list(t.shape) for t in
                        (lhs, rhs, scale_a, scale_b, out)],
                    "padding_factor": 128, "native_calls": 0, "synchronized": False}
        calls.append(evidence)
        (output / "native.json").write_text(json.dumps(calls, indent=2))
        print("ORIGINAL_V5_ENTERED", flush=True)
        binding.call(args, record, a.device)
        evidence["native_calls"] = 1
        print("ORIGINAL_V5_HOST_RETURNED", flush=True)
        result = out[:, ::128]
        result = (result.T if swap else result).contiguous()
        torch.npu.synchronize()
        evidence["synchronized"] = True
        (output / "native.json").write_text(json.dumps(calls, indent=2))
        (output / "native_maps.txt").write_text(Path("/proc/self/maps").read_text())
        cleanup()
        return result

    return Candidate(Implementation(run), "ops-transformer::aclnnGroupedMatmulV5",
                     "Single original full GEMM. Zero-row storage padding maps each weight row "
                     "to its own N128 scale block; operand exchange and output extraction only. "
                     "128x native output expansion; all copying and padding inside run.",
                     pins, limitations=("Padding has substantial memory and compute cost.",),
                     timing_mode="host_roundtrip", cleanup=cleanup)
