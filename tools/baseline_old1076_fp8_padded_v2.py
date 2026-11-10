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
HERE = ROOT / 'reports/baseline-single-agent-local-20261027'
sys.path[:0] = [str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src'), str(ROOT)]
import torch
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import UnsupportedCase
from benchmarks.api import Implementation

spec = importlib.util.spec_from_file_location('original_fp8_padded', HERE/'fp8_padded_adapter.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
output = Path(sys.argv[sys.argv.index('--out')+1])
pins = json.loads((HERE/'FP8_RUNTIME_PREFLIGHT.json').read_text())['source_sha256']
# The historic checker is evidence for historic results, not the checker used now.
pins = {n:h for n,h in pins.items() if not n.endswith('/TileOPs/scripts/ascend_pair_check.py')}
for name, digest in pins.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
pins[str(Path(__file__).resolve())] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

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
            aa=scales(sa,a.shape[0],k); bb=scales(sb,b.shape[0],k)
            result=torch.empty((a.shape[0],b.shape[0]),device=a.device,dtype=getattr(torch,case.params["out_dtype"]))
            for start in range(0,a.shape[0],128):
                end=min(start+128,a.shape[0])
                result[start:end]=native.run(a[start:end],b,aa[start:end],bb)
            return result
        return dataclasses.replace(candidate, implementation=Implementation(run), adaptation=candidate.adaptation+' Exact scale storage expansion and independent M-row chunks of128 only; complete original K reduction per output.')
    return build

registry.candidate_factory = factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
