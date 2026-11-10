"""Pure input/output layout lowering to the existing original full V5 binding."""
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
from integration700_transformer_dense_gmm import dense

def conv1d(case):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    assert case.id == 'whisper-conv1-float16'
    assert (case.params['stride'],case.params['padding'],case.params['dilation'],case.params['groups']) == (1,0,1,1)
    proxy=SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name='GemmFwdOp')),params={'trans_a':False,'trans_b':False})
    inner=dense(proxy)
    def run(x,weight,bias=None):
        assert bias is None
        n,ci,l=x.shape;co,_,k=weight.shape
        lhs=x.unfold(2,k,1).permute(0,2,1,3).contiguous().reshape(n*(l-k+1),ci*k)
        rhs=weight.reshape(co,ci*k).T.contiguous()
        out=inner.implementation.run(lhs,rhs)
        return out.reshape(n,l-k+1,co).transpose(1,2).contiguous()
    return replace(inner,implementation=Implementation(run),adaptation='Complete original V5 no-group API; only unfold, transpose, contiguous pack and output reshape for original unpadded stride-one Conv1d.',source_sha256=inner.source_sha256|source_hashes(Path(__file__)))
