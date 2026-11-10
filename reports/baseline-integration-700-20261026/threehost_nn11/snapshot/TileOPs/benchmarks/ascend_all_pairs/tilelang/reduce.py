"""Exact extrema via repeated calls to the original fixed-size native reducer."""
from math import prod
from pathlib import Path
from .microblocks import _load_builder


def extrema(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    op=case._call.signature.name
    x,=case.inputs
    boolean=op in ('AllFwdOp','AnyFwdOp')
    if boolean and x.dtype!=torch.bool:
        raise UnsupportedCase('Original extrema on 0/1 supports boolean inputs only; no native nonzero conversion is substituted.')
    dtype='int32' if boolean else 'float32'
    nativeop='min' if op in ('AminFwdOp','AllFwdOp') else 'max'
    if not boolean and x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise UnsupportedCase('Floating extrema route only uses lossless FP32 widening.')
    dims=case.params.get('dim')
    dims=tuple(range(x.ndim)) if dims is None else tuple(d%x.ndim for d in ((dims,) if isinstance(dims,int) else dims))
    outer=tuple(d for d in range(x.ndim) if d not in dims)
    order=outer+dims
    m,k=prod(x.shape[d] for d in outer),prod(x.shape[d] for d in dims)
    shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim)) if case.params.get('keepdim',False) else tuple(x.shape[d] for d in outer)
    tl,builder,hashes=_load_builder('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel')
    # Original regression exposes rows/extent shape parameters. This exact
    # (2,128) shape has 256 threads and is exercised by its own test matrix.
    hashes.update(source_hashes(Path(__file__)))
    kernel=tl.compile(builder(2,128,dtype,nativeop),target='ascend',pass_configs={tl.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE:False})
    neutral=(1 if nativeop=='min' else 0) if boolean else (float('inf') if nativeop=='min' else -float('inf'))
    def run(x):
        data=x.permute(order).contiguous().reshape(m,k).to(getattr(torch,dtype))
        while True:
            width=data.shape[-1]
            packed=F.pad(data,(0,-width%128),value=neutral).reshape(-1,128)
            count=packed.shape[0]
            packed=F.pad(packed,(0,0,0,count%2),value=neutral)
            result=torch.empty(packed.shape[0],device=x.device,dtype=data.dtype)
            for start in range(0,packed.shape[0],2):kernel(packed[start:start+2],result[start:start+2])
            data=result[:count].reshape(m,-1)
            if data.shape[-1]==1:break
        return data.reshape(shape).to(torch.bool if boolean else x.dtype)
    return Candidate(Implementation(run),'tilelang::make_default_layout_reduce_kernel(2,128)',
                     'Original 2x128 native extrema, unchanged schedule/pass configuration; exact widening, reduction-axis packing and neutral padding. Repeated original min/max calls combine partial extrema; no Torch reduction. Boolean all/any select min/max on exact 0/1 representations.',hashes)
