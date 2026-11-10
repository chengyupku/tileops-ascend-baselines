"""Expand scale storage into the verified original full GroupedMatmulV5 API."""
import dataclasses
import hashlib
import importlib.util
import json
import runpy
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'reports/baseline-retest32-20261030/server1/deployments/fp8_v64'
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import UnsupportedCase,source_hashes
from benchmarks.api import Implementation
import torch
spec=importlib.util.spec_from_file_location('server1_original_fp8_padded',HERE/'fp8_padded_adapter.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
output=Path(sys.argv[sys.argv.index('--out')+1])
files=[Path(__file__),HERE/'fp8_padded_adapter.py',ROOT/'TileOPs/benchmarks/ascend_math_more.py']
files.extend(p for p in (HERE/'ops-transformer-vendor').rglob('*') if p.is_file() and p.suffix in ('.so','.cpp','.h','.py','.json'))
files.extend(p for p in (ROOT/'ascend_baselines/ops-transformer/gmm/grouped_matmul').rglob('*') if p.is_file())
pins=source_hashes(*files)

def scales(scale, rows, k):
    if scale.numel() == 1:
        return scale.reshape(1, 1).expand(rows, k//128)
    if tuple(scale.shape) == (rows, k//128):
        return scale
    if tuple(scale.shape) == ((rows+127)//128, k//128):
        return scale.repeat_interleave(128, dim=0)[:rows]
    raise UnsupportedCase('No exact storage expansion for this FP32 scale layout')

def factory(case, library):
    def build():
        if library != 'ops-transformer' or case._call.signature.name != 'GemmFP8FwdOp' or len(case.inputs) != 4:
            raise UnsupportedCase('Bias-free original FP8 GEMM only')
        a,b,sa,sb = case.inputs
        k=a.shape[1]
        if k%128:
            raise UnsupportedCase('Original K128 scaling requires aligned complete K')
        normalized = types.SimpleNamespace(inputs=(a,b,scales(sa,a.shape[0],k),scales(sb,b.shape[0],k)), params=case.params)
        candidate = original.factory(normalized, output, pins)
        native = candidate.implementation
        def run(a,b,sa,sb):
            rows=a.shape[0]; aa=scales(sa,rows,k); bb=scales(sb,b.shape[0],k)
            if rows==1:
                packed=torch.zeros((2,k),dtype=torch.uint8,device=a.device)
                packed[0].copy_(a[0].contiguous().view(torch.uint8))
                a=packed.view(a.dtype); aa=aa.expand(2,k//128)
            return native.run(a,b,aa,bb)[:rows]
        return dataclasses.replace(candidate, implementation=Implementation(run), adaptation=candidate.adaptation+' Exact scale storage expansion only; no scale arithmetic or K partition.')
    return build

registry.candidate_factory = factory
