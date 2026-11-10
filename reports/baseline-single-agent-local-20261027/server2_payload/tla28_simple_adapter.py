"""Minimal original TopK and two-sequence CausalConv bindings."""
import hashlib
import json
from pathlib import Path


def factory(case, native, tilelang, output, identity):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    p = case.params
    topk = 'index_score' in p
    if topk:
        src = 'examples/topk_selector/example_topk_selector.py'
        mod = native.load(src, 'simple_topk_selector')
        b, s, n, g = p['index_score'][0]
        k = p['topk']
        name = 'simple_topk_selector'
        args, kwargs = (b*s*g, n, k, k//4), {}
    else:
        src = 'examples/causal_conv1d/causal_conv1d_pto.py'
        batch, dim, hist = p['conv_state'][0]
        if batch != 2 or hist != 3 or p.get('activation') != 'silu':
            raise UnsupportedCase('Original prefill has two cache-index slots; do not split or rewrite it.')
        mod = native.load(src, '_build_kernel', 'CORE_NUM')
        name = '_build_kernel'
        args, kwargs = (hist+1, mod.CORE_NUM, batch, dim), {'dtype_str': p['x'][1]}
    builder = getattr(mod, name)
    config = builder.__jit_impl__
    kernel = tilelang.compile(builder.__wrapped__(*args, **kwargs), out_idx=config.out_idx,
        workspace_idx=config.workspace_idx, target='pto', platform='A5',
        pass_configs=config.pass_configs, execution_backend=config.execution_backend,
        target_host=config.target_host, compile_flags=config.compile_flags)
    pins = dict(identity)
    for path in (Path(__file__), native.VENDOR / src):
        pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'binding.json').write_text(json.dumps({'api':src+':'+name, 'params':p,
        'native_calls':1, 'original_pass_configs':str(config.pass_configs)}, indent=2))

    def launch(*args):
        torch.npu.synchronize()
        (output / 'PRE_SYNC_COMPLETED.json').write_text('{"synchronized":true}')
        lib = Path(kernel.adapter.lib._name).resolve()
        (output / 'actual_kernel.cu').write_text(kernel.get_kernel_source())
        (output / 'ACTUAL_LAUNCH.json').write_text(json.dumps({'library':str(lib),
            'sha256':hashlib.sha256(lib.read_bytes()).hexdigest(), 'native_calls':1}))
        result = kernel(*args)
        (output / 'HOST_RETURNED.json').write_text('{"returned":true}')
        torch.npu.synchronize()
        (output / 'DEVICE_SYNCHRONIZED.json').write_text('{"synchronized":true}')
        return result

    if topk:
        def run(scores, starts, ends):
            if not (bool((starts == 0).all().item()) and bool((ends == n).all().item())):
                raise UnsupportedCase('Original API has no per-row selection window.')
            x = scores.permute(0, 1, 3, 2).contiguous().reshape(b*s*g, n)
            return launch(x).reshape(b, s, g, k)
    else:
        def run(x, weight, state, cu, bias=None, has_initial_state=None):
            if bias is not None:
                raise UnsupportedCase('Original prefill has no bias input.')
            # Preserve the complete native state result while leaving inputs unchanged.
            ss = state.transpose(1, 2).contiguous().clone()
            idx = torch.arange(batch, dtype=torch.int32, device=x.device)
            flags = (torch.zeros(batch, dtype=torch.int32, device=x.device) if has_initial_state is None
                     else has_initial_state.to(torch.int32).contiguous())
            # Original four-token loads may read three unused tail rows.
            xx = F.pad(x.T.contiguous(), (0, 0, 0, 3))
            yy = launch(xx, weight.T.contiguous(), ss, cu, idx, flags)
            return yy[:x.shape[1]].T.contiguous(), ss.transpose(1, 2).contiguous()
    return Candidate(Implementation(run), 'tilelang-ascend::'+src+':'+name,
        'Original full kernel; layout, tail padding and private state only; no precision or tiling change.',
        pins, timing_mode='host_roundtrip')
