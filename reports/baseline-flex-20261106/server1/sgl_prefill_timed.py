from pathlib import Path
import hashlib
from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
from benchmarks.ascend_attention_cases import _contract
import benchmarks.ascend_attention_cases as original_adapter

def build(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    b, sq, h, d, sk, hk, dv, scale = _contract(case)
    if not case.params.get('is_causal') or sq <= 1 or d != dv or d != 128:
        raise ValueError('Exact causal equal-feature prefill required')
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.attention.sinks_attention', 'attention_sinks_prefill_triton', host)
    pages = (sk + 127) // 128
    def run(q, k, v, *extra):
        kk = F.pad(k, (0,0,0,0,0,pages*128-sk)).contiguous().reshape(b*pages,128,hk,d)
        vv = F.pad(v, (0,0,0,0,0,pages*128-sk)).contiguous().reshape(b*pages,128,hk,d)
        table = torch.arange(b*pages, device=q.device, dtype=torch.int32).reshape(b,pages)
        lengths = torch.full((b,), sk, device=q.device, dtype=torch.int32)
        queries = torch.full((b,), sq, device=q.device, dtype=torch.int32)
        sinks = torch.full((h,), float('-inf'), device=q.device, dtype=torch.float32)
        return fn(q.reshape(b*sq,h*d).contiguous(), kk, vv, sinks, queries, table, lengths, scale, -1, h, hk).reshape(b,sq,h,d)
    hashes = source_hashes(Path(original_adapter.__file__), LIBRARIES/'sgl-kernel-npu'/host)
    hashes[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run), 'sgl_kernel_npu.attention.sinks_attention.attention_sinks_prefill_triton',
        'One original batched causal prefill API; negative-infinity sink removes sink mass. All page layout, metadata and output view inside timed run; original native source unchanged.', hashes, timing_mode='host_roundtrip')


def build_rope(case):
    from benchmarks.ascend_rope_attention_cases import _with_rope
    return _with_rope(case, 'sgl-kernel-npu', build)
