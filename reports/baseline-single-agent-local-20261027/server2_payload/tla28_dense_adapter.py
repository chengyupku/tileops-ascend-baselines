"""Minimal whole-batch binding to the original FP16 GQA example."""
import hashlib
import json
from pathlib import Path


def factory(case, native, tilelang, output, identity):
    import torch
    import torch.nn.functional as F
    from tilelang.intrinsics import make_zn_layout, make_nz_layout
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    p = case.params
    if p['q'][0][-1] > 128:
        from gqa_full_flash_adapter import factory as full_flash
        return full_flash(case, native, tilelang, output, identity)
    batch, sq, heads, dim = p['q'][0]
    sk, kh = p['k'][0][1:3]
    if (p['q'][1] != 'float16' or dim != 128
            or p.get('window_size_left', -1) >= 0 or p.get('window_size_right', -1) >= 0
            or p.get('softcap') not in (None, 0) or p.get('pos_encoding_mode', 'none') != 'none'):
        raise UnsupportedCase('Original FP16 D128 full GQA API contract.')
    src = 'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py'
    mod = native.load(src, 'gqa_sink_fwd', 'make_attention_mask')
    mod.make_zn_layout, mod.make_nz_layout = make_zn_layout, make_nz_layout
    builder, config = mod.gqa_sink_fwd, mod.gqa_sink_fwd.__jit_impl__
    mq, nk = (sq + 63) // 64 * 64, (sk + 127) // 128 * 128
    causal = p.get('is_causal', False)
    kernel = tilelang.compile(builder.__wrapped__(batch, heads // kh, heads, dim, mq, nk, causal,
            sm_scale=p.get('sm_scale')), out_idx=config.out_idx, workspace_idx=config.workspace_idx,
            target='pto', platform='A5', pass_configs=config.pass_configs,
            execution_backend=config.execution_backend, target_host=config.target_host,
            compile_flags=config.compile_flags)
    pins = dict(identity)
    for path in (Path(__file__), native.VENDOR / src):
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'binding.json').write_text(json.dumps({'api': src + ':gqa_sink_fwd',
        'params': p, 'native_calls': 1, 'original_pass_configs': str(config.pass_configs)}, indent=2))

    def run(q, k, v, *optional):
        if any(x is not None for x in optional):
            raise UnsupportedCase('Optional scale/position arithmetic is not implemented in binding.')
        qp = F.pad(q.transpose(1, 2).contiguous(), (0, 0, 0, mq-sq))
        kp = F.pad(k.transpose(1, 2).contiguous(), (0, 0, 0, nk-sk))
        vp = F.pad(v.transpose(1, 2).contiguous(), (0, 0, 0, nk-sk))
        ql = torch.full((batch,), sq, dtype=torch.int32, device=q.device)
        kl = torch.full((batch,), sk, dtype=torch.int32, device=q.device)
        mask, _ = mod.make_attention_mask(batch, mq, nk, ql, kl, causal, 64, 128, q.device)
        sinks = torch.full((heads,), float('-inf'), dtype=q.dtype, device=q.device)
        torch.npu.synchronize()
        (output / 'PRE_SYNC_COMPLETED.json').write_text('{"synchronized": true}')
        lib = Path(kernel.adapter.lib._name).resolve()
        (output / 'actual_kernel.cu').write_text(kernel.get_kernel_source())
        (output / 'ACTUAL_LAUNCH.json').write_text(json.dumps({'library': str(lib),
            'sha256': hashlib.sha256(lib.read_bytes()).hexdigest(), 'native_calls': 1}))
        result = kernel(qp, kp, vp, sinks, ql, kl, nk, mask)
        (output / 'HOST_RETURNED.json').write_text('{"returned": true}')
        torch.npu.synchronize()
        (output / 'DEVICE_SYNCHRONIZED.json').write_text('{"synchronized": true}')
        return result[:, :, :sq].transpose(1, 2).contiguous()
    return Candidate(Implementation(run), 'tilelang-ascend::' + src + ':gqa_sink_fwd',
        'Original whole-batch GQA; tensor layout/padding and original mask helper only.',
        pins, timing_mode='host_roundtrip')
