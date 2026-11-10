"""Private SGL GDN/DeltaNet adapter for exact-scope validation.
Uses the inspected original chunk_gated_delta_rule_native API only.
"""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes

ROOT = LIBRARIES / 'sgl-kernel-npu'
HOST = 'python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk.py'

def _native():
    return local_function('sgl-kernel-npu', 'python/sgl_kernel_npu',
                          'sgl_kernel_npu.fla.chunk_gated_delta_rule_npu', 'chunk_gated_delta_rule_npu', 'python/sgl_kernel_npu/sgl_kernel_npu/fla/chunk_gated_delta_rule_npu.py')

def _dense_delta(case):
    import torch
    from benchmarks.api import Implementation
    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    p = case.params
    if offsets is not None or offsets_cpu is not None:
        raise UnsupportedCase('Private representative adapter is dense only; packed case not probed yet.')
    if p.get('use_qk_l2norm_in_kernel', False):
        raise UnsupportedCase('Native route here is only the no-L2 contract.')
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or beta.dtype != q.dtype:
        raise UnsupportedCase('Native SGL route requires BF16 Q/K/V/beta.')
    fn = _native()
    b, s, h, dk = q.shape
    hv, dv = v.shape[2:]
    scale = p.get('scale')
    if scale is not None and scale != dk ** -0.5:
        raise UnsupportedCase('Native API has fixed scale=1/sqrt(K); non-default scale is out of scope.')
    gate = torch.zeros((b, s, hv), device=q.device, dtype=q.dtype)
    # Native original API consumes [B,S,H,*], maintains FP32 state, and returns
    # output plus final_state. It repeats Q/K along heads for GVA as the API does.
    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        state = None if initial is None else initial.contiguous()
        out, final = fn(q.contiguous().view(b*s, h, dk), k.contiguous().view(b*s, h, dk),
           v.contiguous().view(b*s, hv, dv), beta=beta.contiguous().view(b*s, hv),
           initial_state=None if state is None else state.transpose(-1,-2).contiguous(),
           actual_seq_lengths=torch.full((b,), s, dtype=torch.int32, device=q.device),
           scale=scale, g=None)
        return out.view(b,s,hv,dv), final.transpose(-1,-2).contiguous()
    files = [ROOT / HOST, Path(__file__), ROOT / 'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py']
    files += sorted((ROOT / 'python/sgl_kernel_npu/sgl_kernel_npu/fla').glob('*.py'))
    return Candidate(Implementation(run),
                     'sgl_kernel_npu.fla.chunk_gated_delta_rule_npu',
                     'Original complete native GDN API; zero gate, FP32 initial/final state, full output tuple; no kernel/schedule changes.',
                     source_hashes(*files),
                     limitations=('Private dense no-L2 probe; packed and l2norm cases require separate exact adapter.',))

def _gdn(case):
    return _dense_delta(case)

def _delta(case):
    import torch
    from benchmarks.api import Implementation
    q, k, v, beta, initial, offsets, offsets_cpu = case.inputs
    p = case.params
    if offsets is not None or offsets_cpu is not None:
        raise UnsupportedCase('Private representative adapter is dense only.')
    if p.get('use_qk_l2norm_in_kernel', False):
        raise UnsupportedCase('Native route here is only the no-L2 contract.')
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or beta.dtype != q.dtype:
        raise UnsupportedCase('Native SGL route requires BF16 Q/K/V/beta.')
    fn = _native(); b, s, h, dk = q.shape; hv, dv = v.shape[2:]
    scale = p.get('scale')
    if scale is not None and scale != dk ** -0.5:
        raise UnsupportedCase('Native API has fixed scale=1/sqrt(K).')
    gate = torch.zeros((b, s, hv), device=q.device, dtype=q.dtype)
    def run(q, k, v, beta, initial, offsets, offsets_cpu):
        out, final = fn(q.contiguous().view(b*s, h, dk), k.contiguous().view(b*s, h, dk),
           v.contiguous().view(b*s, hv, dv), beta=beta.contiguous().view(b*s, hv),
           initial_state=None if initial is None else initial.transpose(-1,-2).contiguous(),
           actual_seq_lengths=torch.full((b,), s, dtype=torch.int32, device=q.device),
           scale=scale, g=None)
        return out.view(b,s,hv,dv), final.transpose(-1,-2).contiguous()
    files = [ROOT / HOST, Path(__file__), ROOT / 'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py']
    files += sorted((ROOT / 'python/sgl_kernel_npu/sgl_kernel_npu/fla').glob('*.py'))
    return Candidate(Implementation(run),
                     'sgl_kernel_npu.fla.chunk_gated_delta_rule_npu',
                     'Original complete native DeltaNet API with zero gate and full state/output tuple.',
                     source_hashes(*files), limitations=('Private dense no-L2 probe.',))

PROVIDERS = {'GDNFwdOp': _gdn, 'DeltaNetFwdOp': _delta}
