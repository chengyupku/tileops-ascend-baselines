from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, local_function, source_hashes
from benchmarks.ascend_attention_cases import _contract

def sgl_gqa_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    batch, sq, heads, dim, sk, kvheads, vdim, scale = _contract(case)
    group = heads // kvheads
    padded_group = 1 << (group - 1).bit_length()
    page_size = 128
    pages = (sk + page_size - 1) // page_size
    host = 'python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn = local_function('sgl-kernel-npu', 'python/sgl_kernel_npu', 'sgl_kernel_npu.attention.decode_attention', 'decode_gqa', host)
    def run(q, k, v, *extra):
        table = torch.arange(batch * pages, device=q.device, dtype=torch.int32).view(batch, pages)
        lengths = [torch.full((batch,), sk - sq + i + 1 if case.params.get('is_causal') else sk,
                              device=q.device, dtype=torch.int32) for i in range(sq)]
        packed_q = F.pad(q.reshape(batch, sq, kvheads, group, dim), (0, 0, 0, padded_group - group))
        packed_q = packed_q.reshape(batch, sq, kvheads * padded_group, dim)
        packed_k = F.pad(k, (0, 0, 0, 0, 0, pages * page_size - sk)).contiguous().view(batch * pages, page_size, kvheads, dim)
        packed_v = F.pad(v, (0, 0, 0, 0, 0, pages * page_size - sk)).contiguous().view(batch * pages, page_size, kvheads, vdim)
        output = torch.empty((batch, sq, kvheads * padded_group, vdim), device=q.device, dtype=q.dtype)
        for i in range(sq):
            fn(packed_q[:, i].contiguous(), packed_k, packed_v, output[:, i], lengths[i], scale, page_size, table)
        return output.reshape(batch, sq, kvheads, padded_group, vdim)[..., :group, :].contiguous().reshape(batch, sq, heads, vdim)
    return Candidate(Implementation(run), 'sgl_kernel_npu.attention.decode_attention.decode_gqa',
                     'Pad independent query heads and final cache page; original length masks select each causal prefix; invoke original decode once per query; crop heads.',
                     source_hashes(Path(__file__), LIBRARIES / 'sgl-kernel-npu' / host),
                     limitations=('All per-call packing and per-query native launches count toward execution time.',))
