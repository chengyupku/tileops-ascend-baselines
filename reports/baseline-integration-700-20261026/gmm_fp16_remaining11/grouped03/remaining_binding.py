"""Layout/dtype-only adapters around the complete original GroupedMatmul V5 API."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from v5_fp32_bias import dense_fp32

def remaining(case):
    import torch, math
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    p=case.params;op=case._call.signature.name
    proxy=SimpleNamespace(_call=SimpleNamespace(signature=SimpleNamespace(name='GemmFwdOp')),params={'trans_a':False,'trans_b':False})
    inner=dense_fp32(proxy);product=inner.implementation.run
    def conv(x,w,bias=None):
        d=x.ndim-2
        tup=lambda v:(v,)*d if isinstance(v,int) else tuple(v)
        st,pad,dil=tup(p['stride']),tup(p['padding']),tup(p['dilation'])
        assert p['groups']==1
        x=torch.nn.functional.pad(x,tuple(t for a in reversed(pad) for t in (a,a)))
        outshape=tuple((x.shape[i+2]-dil[i]*(w.shape[i+2]-1)-1)//st[i]+1 for i in range(d))
        shape=(x.shape[0],x.shape[1])+outshape+tuple(w.shape[2:])
        strides=x.stride()[:2]+tuple(x.stride()[i+2]*st[i] for i in range(d))+tuple(x.stride()[i+2]*dil[i] for i in range(d))
        windows=x.as_strided(shape,strides)
        lhs=windows.permute((0,)+tuple(range(2,2+d))+(1,)+tuple(range(2+d,2+2*d))).contiguous().reshape(-1,w[0].numel())
        rhs=w.reshape(w.shape[0],-1).T.contiguous()
        y=product(lhs,rhs,bias)
        return y.reshape((x.shape[0],)+outshape+(w.shape[0],)).movedim(-1,1).contiguous()
    def grouped(a,b,sizes,offsets=None):
        outputs=[];start=0
        for i,n in enumerate(sizes.tolist()):
            end=start+n
            if p['transpose_a']:
                lhs=a[start:end].T;rhs=b[:,start:end].T if p['transpose_b'] else b[start:end]
            else:lhs=a[start:end];rhs=b[i].T if p['transpose_b'] else b[i]
            outputs.append(product(lhs.contiguous(),rhs.contiguous()));start=end
        return torch.stack(outputs) if p['transpose_a'] else torch.cat(outputs)
    def moe(a,b,metadata):
        assert p['activation'] is None and p['out_dtype'] is None
        assert p['layout']=={'contiguous':{'packing':'tight','metadata_kind':'physical_psum','alignment':1}}
        out=torch.zeros((*a.shape[:-1],b.shape[1]),dtype=a.dtype,device=a.device);start=0
        for i,end in enumerate(metadata.tolist()):
            if end>start:out[start:end]=product(a[start:end].contiguous(),b[i].T.contiguous())
            start=end
        return out
    def hadamard(x):
        assert p['base_order']==1 and x.shape[-1]==4096
        order=64;radix=64;n=4096
        signs=lambda d:[[(-1. if (i&j).bit_count()%2 else 1.) for j in range(d)] for i in range(d)]
        hradix=torch.tensor(signs(radix),dtype=torch.float32,device=x.device)
        first=product(x.float().reshape(-1,radix),hradix)
        normalized=[[value/math.sqrt(n) for value in row] for row in signs(order)]
        hbase=torch.tensor(normalized,dtype=torch.float32,device=x.device).T.contiguous()
        packed=first.reshape(-1,order,radix).transpose(1,2).contiguous().reshape(-1,order)
        second=product(packed,hbase).reshape(-1,radix,order).transpose(1,2).contiguous()
        return second.reshape_as(x).to(x.dtype)
    run=conv if op.startswith('Conv') else grouped if op=='GroupedGemmFwdOp' else moe if op=='MoEGroupedGemmFwdOp' else hadamard
    return replace(inner,implementation=Implementation(run),source_sha256=inner.source_sha256|source_hashes(Path(__file__)),adaptation='Complete original V5 FP32 API with lossless input widening/output cast; native optional bias. Layout-only convolution window packing or metadata group slicing; existing fixed Hadamard coefficient convention.')
