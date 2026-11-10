"""Original ops-math Mul, with broadcast scale layout instead of materialization."""
from types import SimpleNamespace
from pathlib import Path
import torch
import torch.nn.functional as F

def factory(op,params,shape):
    import benchmarks
    path='/opt/tileops-all-pairs/workspace/TileOPs/benchmarks'
    if path not in benchmarks.__path__:benchmarks.__path__.append(path)
    from benchmarks.ascend_all_pairs.ops_math.graph import make
    m,k=shape
    output_shape=(m,(k+127)//128,128) if op=='INT8DequantPerBlockFwdOp' else shape
    child=make(SimpleNamespace(params={'output':(output_shape,'float32')}),'MulFwdOp')
    def run(q,scale):
        qf=q.float()
        if op=='INT8DequantPerBlockFwdOp':
            pad=(-k)%128;qf=(F.pad(qf,(0,pad)) if pad else qf).reshape(output_shape)
            scale=scale[:,:,None]
        elif op=='INT8DequantPerChannelFwdOp':scale=scale[:,None]
        y=child.implementation.run(qf,scale)
        y=y.reshape(m,-1)[:,:k].to(getattr(torch,params['out_dtype']))
        child.cleanup()
        return y
    return run,child.source_sha256
