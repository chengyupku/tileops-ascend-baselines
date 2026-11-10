"""Equivalent eager expressions; every conversion stays inside the timed function."""
import torch

def int4_quant_per_group(w, group_size):
    n,k=w.shape
    groups=w.float().view(-1,group_size)
    lo,hi=groups.amin(dim=-1),groups.amax(dim=-1)
    multiplier=torch.where(hi==lo,1.0,16.0/(hi-lo))
    offset=(hi+lo)*0.5
    q=((groups-offset[:,None])*multiplier[:,None]).round().clamp(-8,7)
    q=q.view(n,k//2,2)
    low=q[...,1]
    packed=(q[...,0]*16+torch.where(low<0,low+16,low)).to(torch.int8)
    return packed,torch.stack((1.0/multiplier,offset),dim=-1)

def complex_truth(x):
    # Mask the sign bit only; preserve subnormal, infinity and NaN truth values.
    dtype,mask=(torch.int32,0x7fffffff) if x.dtype==torch.complex64 else (torch.int64,0x7fffffffffffffff)
    words=x.contiguous().view(dtype).reshape(*x.shape,2)
    nonzero=(words & mask)!=0
    return nonzero[...,0] | nonzero[...,1]

def reduction(name,x,p):
    truth=complex_truth(x)
    if name=='CountNonzeroFwdOp':return torch.count_nonzero(truth,dim=p.get('dim')).to(torch.int64)
    fn=torch.all if name=='AllFwdOp' else torch.any
    return fn(truth,dim=p.get('dim'),keepdim=p.get('keepdim',False))
