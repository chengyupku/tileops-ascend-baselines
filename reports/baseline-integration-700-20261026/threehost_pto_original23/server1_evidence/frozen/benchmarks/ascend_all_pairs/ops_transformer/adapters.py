"""Layout-only bindings to unmodified ops-transformer implementations."""
from pathlib import Path


def bmm(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from dataclasses import replace

    a, b = case.inputs
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise UnsupportedCase('GroupedMatmulAdd native inputs require matching FP16/BF16.')
    binding = GraphCall('ops-transformer')
    def run(a, b):
        # Every slice, transpose and accumulator initialization belongs to run.
        outputs = []
        for lhs, rhs in zip(a, b, strict=True):
            group = torch.full((1,), lhs.shape[1], device=a.device, dtype=torch.int64)
            out = torch.zeros((1, lhs.shape[0], rhs.shape[1]), device=a.device, dtype=torch.float32)
            binding.call('GroupedMatmulAdd', [lhs.T.contiguous(), rhs.contiguous(), group, out], [out],
                         {'transpose_x': True, 'transpose_weight': False,
                          'group_type': 2, 'group_list_type': 0})
            outputs.append(out[0].to(a.dtype))
        return torch.stack(outputs)
    result = _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add',
                        'One native K-group per independent batch; per-call transpose, zero FP32 accumulator, manifest output cast and stack. No input-dependent caching.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))
