"""One original full FlashAttention API for single-query FP16 GQA.

Repeat KV heads, pad unused query rows, and crop the output. Every query row
retains the full key extent. No query/key splitting or external attention math.
The original source contains explicit barriers; its passes remain unchanged.
"""
import hashlib
import json
from pathlib import Path


def factory(case, native, tilelang, output, identity):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    p = case.params
    batch, queries, heads, dim = p["q"][0]
    length, kv_heads = p["k"][0][1:3]
    if (queries != 1 or length % 64 or heads % kv_heads
            or p["q"][1] != "float16"
            or p.get("window_size_left", -1) >= 0
            or p.get("window_size_right", -1) >= 0
            or p.get("softcap", 0) not in (0, None)
            or p.get("pos_encoding_mode", "none") not in ("none", None)):
        raise UnsupportedCase("Full original noncausal FP16 API requires an unwindowed single query.")
    source = "examples/flash_attention/flash_attn_bhsd.py"
    module = native.load(source, "flash_attention_fwd")
    module.DataType = tilelang.DataType
    builder = module.flash_attention_fwd
    config = builder.__jit_impl__
    kernel = tilelang.compile(builder.__wrapped__(batch, length, heads, dim),
        out_idx=config.out_idx, workspace_idx=config.workspace_idx,
        target="pto", platform="A5", pass_configs=config.pass_configs,
        execution_backend=config.execution_backend, target_host=config.target_host,
        compile_flags=config.compile_flags)
    pins = dict(identity)
    for path in (Path(__file__), native.VENDOR / source):
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "binding.json").write_text(json.dumps({
        "api": source + ":flash_attention_fwd", "batch": batch, "length": length,
        "heads": heads, "dim": dim, "query_padding_factor": length,
        "original_pass_configs": str(config.pass_configs),
        "original_workspace_indices": config.workspace_idx}, indent=2))

    def run(q, k, v, *optional):
        if any(x is not None for x in optional):
            raise UnsupportedCase("No scale/bias or other optional arithmetic in this binding.")
        qp = F.pad(q.transpose(1, 2).contiguous(), (0, 0, 0, length - 1))
        kp = k.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2).contiguous()
        vp = v.repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2).contiguous()
        torch.npu.synchronize()
        (output / "PRE_SYNC_COMPLETED.json").write_text('{"synchronized": true}\n')
        actual = Path(kernel.adapter.lib._name).resolve()
        code = output / "actual_kernel.cu"
        code.write_text(kernel.get_kernel_source())
        for path in (actual, code):
            pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "ACTUAL_LAUNCH.json").write_text(json.dumps({
            "library": str(actual), "sha256": pins[str(actual)],
            "target": str(kernel.target), "native_calls": 1}, indent=2))
        result = kernel(qp, kp, vp)
        (output / "HOST_RETURNED.json").write_text('{"returned": true}\n')
        result = result[:, :, :1].transpose(1, 2).contiguous()
        torch.npu.synchronize()
        (output / "DEVICE_SYNCHRONIZED.json").write_text('{"synchronized": true}\n')
        return result

    return Candidate(Implementation(run), "tilelang-ascend::" + source + ":flash_attention_fwd",
                     "One complete original FlashAttention; repeated KV heads and padded "
                     "query rows with output cropping. Full batch and key extent in one call. "
                     "All copies inside run; substantial unused-query computation is included.",
                     pins, limitations=("FP16, single query, no optional scales or windows.",),
                     timing_mode="host_roundtrip")
