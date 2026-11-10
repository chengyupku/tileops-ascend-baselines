"""Private top-level adapter. Import/build/run only inside host npu-guard.
Not an acceptance record; fixed to audited FP32 case until full checker passes.
"""
import hashlib
from pathlib import Path

BUILD = Path('/opt/baseline-flex-20261106/server2/ops-math-build-v1')
EXPECTED = {
 'add/libops_math_add.so': '92783955e9707ab8af91987139437f6dd17a37bf1de728bf9e9c62a1def011f7',
 'layernorm/libops_math_layernorm.so': 'ac080fb931f8aa41d59a56871fbcdc90a2c17cd5150f9e62c3803946a1feb6b2',
}

def make(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    x,residual,weight,bias=case.inputs
    if tuple(x.shape)!=(2048,5120) or any(t.dtype!=torch.float32 for t in case.inputs):
        raise UnsupportedCase('Only audited FP32 [2048,5120] case is enabled.')
    if residual.shape!=x.shape or tuple(weight.shape)!=(5120,) or bias.shape!=weight.shape:
        raise UnsupportedCase('Shape contract mismatch.')
    eps=case.params.get('eps',1e-6)
    if eps!=1e-6:
        raise UnsupportedCase('Only current eps=1e-6 contract is enabled.')
    hashes={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for relative,expected in EXPECTED.items():
        p=BUILD/relative
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=expected: raise RuntimeError('Native binary identity changed: '+str(p))
        hashes[str(p)]=actual
        torch.ops.load_library(str(p))
    def run(x,residual,weight,bias):
        summed=torch.ops.allpairs_math_example.add(x.contiguous(),residual.contiguous())
        output=torch.ops.allpairs_math_example.layer_norm(summed,weight.contiguous(),bias.contiguous(),eps)
        return output,summed
    return Candidate(Implementation(run),'ops-math::original fast-launch Add + LayerNorm',
                     'Two unchanged native calls; FP32 residual rounded before norm; dual outputs; packing and allocation remain inside run.',
                     hashes,limitations=('Full current checker and v3 raw acceptance pending.',))
