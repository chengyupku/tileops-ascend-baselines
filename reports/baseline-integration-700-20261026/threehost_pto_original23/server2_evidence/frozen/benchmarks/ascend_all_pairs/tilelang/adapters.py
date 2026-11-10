"""Manifest adapters to unchanged vendor TileLang kernels."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def bmm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_gemm_cases import tilelang_gemm_cases
    a, b = case.inputs
    proxy = SimpleNamespace(inputs=(a[0], b[0]), params={"trans_a": False, "trans_b": False})
    native = tilelang_gemm_cases(proxy)
    def run(a, b):
        return torch.stack([native.implementation.run(a[i], b[i]) for i in range(a.shape[0])])
    return replace(native, implementation=Implementation(run),
                   adaptation=native.adaptation + ' Batch slicing and output stacking on every invocation; unchanged original GEMM per batch.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
