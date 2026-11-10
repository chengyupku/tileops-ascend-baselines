"""Top-level equivalent layouts; no baseline kernel or checker modification."""
import torch
import torch.nn.functional as F
from workloads.quantization import quantize as ref

def int8_block(x):
    m,k=x.shape
    xf=x.float()
    pad=(-k)%128
    blocks=(F.pad(xf,(0,pad)) if pad else xf).reshape(m,(k+127)//128,128)
    scale=ref._int8_scale(blocks.abs().amax(-1))
    q=ref._to_int8(blocks,scale[:,:,None]).reshape(m,-1)[:,:k]
    return q,scale

def fp8_block(w):
    n,k=w.shape
    wf=w.float();pn=(-n)%128;pk=(-k)%128
    padded=F.pad(wf,(0,pk,0,pn)) if pn or pk else wf
    tiles=padded.reshape((n+pn)//128,128,(k+pk)//128,128)
    am=tiles.abs().amax((1,3))
    scale=torch.where(am>0,am/448,torch.ones_like(am))
    q=(tiles/scale[:,None,:,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    return q.reshape(n+pn,k+pk)[:n,:k],scale

def dequant_block(q,scale,out_dtype):
    m,k=q.shape;pad=(-k)%128
    qf=q.float();padded=F.pad(qf,(0,pad)) if pad else qf
    y=padded.reshape(m,(k+127)//128,128)*scale[:,:,None]
    return y.reshape(m,-1)[:,:k].to(out_dtype)

def int4_group(w,group_size):
    n,k=w.shape;groups=w.float().reshape(-1,group_size)
    lo,hi=groups.amin(-1),groups.amax(-1)
    multiplier=torch.where(hi==lo,1.0,16.0/(hi-lo));offset=(hi+lo)*0.5
    q=((groups-offset[:,None])*multiplier[:,None]).round().clamp(-8,7).reshape(n,k//2,2)
    low=q[...,1]
    packed=(q[...,0]*16+torch.where(low<0,low+16,low)).to(torch.int8)
    return packed,torch.stack((1.0/multiplier,offset),-1)

def factory(op,params,reference):
    if op=='INT8QuantPerBlockFwdOp':return int8_block
    if op=='FP8QuantPerBlockFwdOp':return fp8_block
    if op=='INT4QuantPerGroupFwdOp':return lambda w:int4_group(w,params['group_size'])
    if op=='INT8DequantPerBlockFwdOp':
        dt=getattr(torch,params['out_dtype']);return lambda q,s:dequant_block(q,s,dt)
    return reference

def native_factory(op,params):
    import torch_npu
    if op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp'):
        dt=getattr(torch,params['out_dtype'])
        if dt==torch.float32:return None
        def run(q,s):
            scale=s[:,None] if op=='INT8DequantPerChannelFwdOp' else s
            return torch_npu.npu_anti_quant(q,scale,offset=None,dst_dtype=dt)
        return run
    return None
