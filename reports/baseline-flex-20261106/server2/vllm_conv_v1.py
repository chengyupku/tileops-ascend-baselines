"""Private group1 im2col adapter; unchanged original vLLM GEMM, one call."""
from pathlib import Path
from math import prod
import hashlib

def make(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_all_pairs.vllm.runtime import module, candidate, TRITON_ROOT
    p=case.params
    if p['groups']!=1: raise ValueError('Only audited group1 route')
    native,sources=module(TRITON_ROOT+'batch_invariant/matmul.py')
    ish=p['input'][0];wsh=p['weight'][0];out=p['output'][0];dims=len(ish)-2
    def axes(v):return [v]*dims if isinstance(v,int) else list(v)
    stride=axes(p['stride']);dilation=axes(p['dilation']);ks=wsh[2:]
    extent=[d*(k-1)+1 for k,d in zip(ks,dilation)]
    if isinstance(p['padding'],str):raise ValueError('Only numeric padding audited')
    padding=[v for n in reversed(axes(p['padding'])) for v in (n,n)]
    m=ish[0]*prod(out[2:]);k=ish[1]*prod(ks)
    def run(x,weight,bias=None):
        if (bias is not None)!=('bias' in p):raise ValueError('bias presence mismatch')
        windows=F.pad(x,padding)
        for axis in range(dims):windows=windows.unfold(axis+2,extent[axis],stride[axis])
        slices=(slice(None),slice(None))+tuple(slice(0,n) for n in out[2:])+tuple(slice(None,None,d) for d in dilation)
        order=[0]+list(range(2,dims+2))+[1]+list(range(dims+2,2*dims+2))
        a=windows[slices].permute(order).contiguous().reshape(m,k)
        b=weight.contiguous().reshape(wsh[0],k).T.contiguous()
        y=native.matmul_persistent(a,b,bias.contiguous() if bias is not None else None)
        return y.reshape(ish[0],*out[2:],wsh[0]).permute([0,dims+1]+list(range(1,dims+1))).contiguous()
    c=candidate(run,'matmul_persistent:group1 im2col','Pad/unfold/dilated view and transpose layout inside run; one original native GEMM with optional original bias epilogue; no torch arithmetic replacement',sources)
    c.source_sha256[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return c
