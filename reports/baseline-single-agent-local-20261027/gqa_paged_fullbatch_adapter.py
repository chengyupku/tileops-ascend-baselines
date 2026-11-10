"""Paged storage to one original complete variable-length GQA API call."""
import hashlib
import json
from pathlib import Path


def factory(case, native, tilelang, output, identity):
    import torch
    from tilelang.intrinsics import make_zn_layout, make_nz_layout
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    p = case.params
    if (p["q"][1] != "float16" or p["k_pages"][1] != "float16"
            or p["v_pages"][1] != "float16" or p["q"][0][-1] != 128
            or p.get("window_size_left", -1) >= 0 or p.get("window_size_right", -1) >= 0
            or p.get("softcap") not in (None, 0)
            or p.get("pos_encoding_mode", "none") != "none"
            or p.get("out_dtype") not in (None, "float16")):
        raise UnsupportedCase("Original full FP16 D128 API without window/softcap/position transform.")
    source = "examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py"
    module = native.load(source, "gqa_sink_fwd", "make_attention_mask")
    module.make_zn_layout = make_zn_layout
    module.make_nz_layout = make_nz_layout
    builder = module.gqa_sink_fwd
    config = builder.__jit_impl__
    pins = dict(identity)
    for path in (Path(__file__), native.VENDOR / source):
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    kernels = {}

    def run(q, k_pages, v_pages, table, lengths, cu_q, *optional):
        if any(x is not None for x in optional):
            raise UnsupportedCase("No optional arithmetic in this binding.")
        bounds, lens = cu_q.tolist(), lengths.tolist()
        batch = len(lens)
        qlens = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
        assert bounds[0] == 0 and bounds[-1] == q.shape[0]
        assert len(qlens) == batch and min(qlens) > 0 and min(lens) > 0
        heads, dim = q.shape[1:]
        page_size, kv_heads = k_pages.shape[1:3]
        mq, nk = (max(qlens) + 63) // 64 * 64, (max(lens) + 127) // 128 * 128
        qp = torch.zeros((batch, heads, mq, dim), dtype=q.dtype, device=q.device)
        kp = torch.zeros((batch, kv_heads, nk, dim), dtype=q.dtype, device=q.device)
        vp = torch.zeros_like(kp)
        # Layout copies only: all sequences retain their complete key extent.
        for b, (start, end, length) in enumerate(zip(bounds[:-1], bounds[1:], lens)):
            qp[b, :, :end-start].copy_(q[start:end].transpose(0, 1))
            ids = table[b, :(length + page_size - 1) // page_size].long()
            kp[b, :, :length].copy_(k_pages[ids].reshape(-1, kv_heads, dim)[:length].transpose(0, 1))
            vp[b, :, :length].copy_(v_pages[ids].reshape(-1, kv_heads, dim)[:length].transpose(0, 1))
        ql = torch.tensor(qlens, dtype=torch.int32, device=q.device)
        kl = lengths.to(torch.int32).contiguous()
        causal = p.get("is_causal", True)
        mask, tiles = module.make_attention_mask(batch, mq, nk, ql, kl, causal, 64, 128, q.device)
        sinks = torch.full((heads,), float("-inf"), dtype=q.dtype, device=q.device)
        key = (batch, mq, nk, heads, kv_heads, dim)
        if key not in kernels:
            kernels[key] = tilelang.compile(
                builder.__wrapped__(batch, heads // kv_heads, heads, dim, mq, nk, causal,
                                    mask_tiles=tiles, sm_scale=p.get("sm_scale")),
                out_idx=config.out_idx, workspace_idx=config.workspace_idx,
                target="pto", platform="A5", pass_configs=config.pass_configs,
                execution_backend=config.execution_backend, target_host=config.target_host,
                compile_flags=config.compile_flags)
        kernel = kernels[key]
        (output / "binding.json").write_text(json.dumps({
            "api": source + ":gqa_sink_fwd", "params": p, "native_calls": 1,
            "batch": batch, "query_lengths": qlens, "key_lengths": lens,
            "original_pass_configs": str(config.pass_configs),
            "original_workspace_indices": config.workspace_idx}, indent=2))
        torch.npu.synchronize()
        (output / "PRE_SYNC_COMPLETED.json").write_text('{"synchronized": true}\n')
        actual = Path(kernel.adapter.lib._name).resolve()
        code = output / "actual_kernel.cu"
        code.write_text(kernel.get_kernel_source())
        for path in (actual, code):
            pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "ACTUAL_LAUNCH.json").write_text(json.dumps({
            "library": str(actual), "sha256": pins[str(actual)], "native_calls": 1}, indent=2))
        result = kernel(qp, kp, vp, sinks, ql, kl, nk, mask)
        (output / "HOST_RETURNED.json").write_text('{"returned": true}\n')
        torch.npu.synchronize()
        (output / "DEVICE_SYNCHRONIZED.json").write_text('{"synchronized": true}\n')
        return torch.cat([result[b, :, :length].transpose(0, 1) for b, length in enumerate(qlens)], 0).contiguous()

    return Candidate(Implementation(run), "tilelang-ascend::" + source + ":gqa_sink_fwd",
                     "One complete original GQA call for all sequences. Pages gathered and queries padded; "
                     "original mask helper and negative-infinity sinks preserve ordinary GQA. "
                     "All copies and preparation inside run; no performance claim.",
                     pins, timing_mode="host_roundtrip")
