"""Original complete RMSNorm API, with the manifest's native BF16 dtype."""
import hashlib
import json
import math
from pathlib import Path


def factory(case, native, tilelang, out, identity):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run

    assert case.id == "qk-norm-head-bfloat16"
    shape, dtype = case.params["x"]
    assert dtype == "bfloat16"
    hidden = math.prod(case.params["normalized_shape"])
    rows = math.prod(shape) // hidden
    eps = case.params.get("eps")
    if eps is None:
        # Original manifest reference evaluates F.rms_norm on x.float().
        eps = torch.finfo(torch.float32).eps
    source = "examples/normalization/rms_norm.py"
    module = native.load(source, "_rms_norm")
    # Keep the previously selected 64x64 public tile arguments and all upstream
    # passes. PTO is the existing supported compiler route, not a compiler edit.
    kernel = tilelang.compile(
        module._rms_norm.__wrapped__(rows, hidden, 64, 64, eps, dtype=dtype),
        out_idx=[1], pass_configs=module.pass_configs, target="pto",
    )
    hashes = dict(identity)
    for path in (Path(__file__), native.VENDOR / source):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "binding.json").write_text(json.dumps({
        "api": source + ":_rms_norm", "rows": rows, "hidden": hidden,
        "dtype": dtype, "eps": eps, "block_M": 64, "block_N": 64,
        "target": "pto", "source_sha256": hashes,
    }, indent=2))

    def run(x, weight=None, bias=None):
        assert weight is None and bias is None
        packed = x.contiguous().reshape(rows, hidden)
        (out / "PRE_SYNC_ENTERED.json").write_text('{"waiting": true}\n')
        torch.npu.synchronize()
        (out / "PRE_SYNC_COMPLETED.json").write_text('{"completed": true}\n')
        result = kernel(packed)
        (out / "HOST_RETURNED.json").write_text('{"returned": true}\n')
        torch.npu.synchronize()
        (out / "DEVICE_SYNCHRONIZED.json").write_text('{"synchronized": true}\n')
        return result.reshape(shape)

    def observed_run(*args):
        # Cache-loaded adapters expose the actual CDLL path even when the
        # fresh-build lib_generator.libpath metadata is absent.
        actual = Path(kernel.adapter.lib._name).resolve()
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        hashes[str(actual)] = digest
        code = out / "actual_kernel.cu"
        code.write_text(kernel.get_kernel_source())
        hashes[str(code)] = hashlib.sha256(code.read_bytes()).hexdigest()
        (out / "ACTUAL_LAUNCH.json").write_text(json.dumps({
            "loaded_library": str(actual), "sha256": digest,
            "kernel_source": str(code), "target": str(kernel.target),
            "state": "about_to_call_original_kernel",
        }, indent=2))
        result = run(*args)
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
        return result

    return Candidate(
        Implementation(observed_run),
        "tilelang-ascend::" + source + ":_rms_norm",
        "Original complete RMSNorm with public BF16 dtype and epsilon; flatten and restore dimensions. No external FP32 widening or added arithmetic.",
        hashes,
    )
