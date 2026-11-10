"""One complete original ReduceSum, with a lossless axis/layout adaptation.

The old trailing-width-11008 route is not replayed. Transpose the complete input
and reduce axis 0; no splitting, partial sums, dtype change, or internal tuning.
"""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

def make(case):
    from benchmarks.ascend_all_pairs.ops_math.graph import make as original
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.api import Implementation
    assert case.id=='mlp-intermediate-float32'
    assert case.params['x']==([2048,11008],'float32') or list(case.params['x'][0])==[2048,11008]
    assert case.params['dim']==-1 and not case.params['keepdim']
    params=dict(case.params);params['x']=([11008,2048],'float32');params['dim']=0
    native=original(SimpleNamespace(params=params),'SumFwdOp')
    def run(x):
        return native.implementation.run(x.transpose(0,1).contiguous())
    return replace(native,implementation=Implementation(run),
                   adaptation='Lossless full-input transpose [2048,11008] -> [11008,2048], original ReduceSum on axis 0. Same FP32 dtype, all elements, all 2048 outputs; no split/partial reduction.',
                   source_sha256=native.source_sha256|source_hashes(Path(__file__)))
