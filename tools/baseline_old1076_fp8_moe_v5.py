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

pins[str(ROOT/'tools/check_baseline_old1076_pair.py')]=hashlib.sha256((ROOT/'tools/check_baseline_old1076_pair.py').read_bytes()).hexdigest()

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
        if library!='ops-transformer' or case._call.signature.name!='MoEGroupedGemmFP8FwdOp' or 'masked' not in case.params['layout']:
            raise UnsupportedCase('Reviewed original full GEMM masked experts only')
        a,sa,b,sb,metadata=case.inputs
        k=a.shape[-1]; dtype=case.params['output'][1]
        normalized=types.SimpleNamespace(inputs=(a[0],b[0],sa[0],scales(sb[0],b.shape[1],k)),params={'out_dtype':dtype})
        candidate=original.factory(normalized,output,pins)
        native=candidate.implementation
        def run(a,sa,b,sb,metadata):
            result=torch.zeros((a.shape[0],a.shape[1],b.shape[1]),device=a.device,dtype=getattr(torch,dtype))
            for expert,count in enumerate(metadata.tolist()):
                if not 0<=count<=a.shape[1]:raise ValueError('Masked count exceeds declared extent')
                weight_scales=scales(sb[expert],b.shape[1],k)
                for start in range(0,count,128):
                    end=min(start+128,count);lhs=a[expert,start:end];lhs_scales=sa[expert,start:end];rows=end-start
                    if rows==1:
                        packed=torch.zeros((2,k),device=a.device,dtype=torch.uint8)
                        packed[0].copy_(lhs[0].contiguous().view(torch.uint8))
                        lhs=packed.view(a.dtype);lhs_scales=lhs_scales.expand(2,k//128)
                    result[expert,start:end]=native.run(lhs,b[expert],lhs_scales,weight_scales)[:rows]
            return result
        return dataclasses.replace(candidate,implementation=Implementation(run),adaptation=candidate.adaptation+' Independent complete expert/M-row GEMMs; original live-row metadata, zero padded output, exact repeated block-scale storage; no K splitting.')
    return build

registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
