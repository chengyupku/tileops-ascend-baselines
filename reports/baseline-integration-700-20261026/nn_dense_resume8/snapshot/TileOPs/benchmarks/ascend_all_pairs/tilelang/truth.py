"""Nonzero semantics through original unsigned masks and native bit operations."""
from pathlib import Path
from math import prod
from types import SimpleNamespace
from dataclasses import replace
from .merge_native import _builder


def _convert():
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import source_hashes
    tl,builder,hashes=_builder()
    kernels={op:tl.compile(builder(op,'int32'),target='ascend') for op in ('vand','vor','vxor')}
    hashes.update(source_hashes(Path(__file__)))
    def truth(x):
        flat=x.contiguous().flatten();n=flat.numel()
        # Complex values are true if either component is nonzero. Bit views
        # preserve FP64 components and avoid any complex-to-real narrowing.
        if flat.dtype in (torch.complex64,torch.complex128,torch.float64,torch.int64):
            count=flat.element_size()//4
            words=flat.view(torch.int32).reshape(-1,count)
            columns=[words[:,j].contiguous() for j in range(count)]
            clear={0,1} if flat.dtype==torch.complex64 else {1,3} if flat.dtype==torch.complex128 else {1} if flat.dtype==torch.float64 else set()
        elif flat.dtype.is_floating_point:
            columns=[flat.float().view(torch.int32)];clear={0}
        else:columns=[flat.to(torch.int32)];clear=set()
        columns=[F.pad(a,(0,-n%64)) for a in columns]
        zero=torch.zeros(64,device=x.device,dtype=torch.int32);one=torch.ones_like(zero);mask=one.view(torch.uint32)
        signless=torch.full_like(zero,0x7fffffff);acc=torch.empty_like(zero);temporary=torch.empty_like(zero);out=torch.empty_like(columns[0])
        for start in range(0,out.numel(),64):
            bits=columns[0][start:start+64]
            if 0 in clear:
                kernels['vand'](bits,signless,zero,mask,acc);bits=acc
            for j,column in enumerate(columns[1:],1):
                other=column[start:start+64]
                if j in clear:
                    kernels['vand'](other,signless,zero,mask,temporary);other=temporary
                kernels['vor'](bits,other,zero,mask,acc);bits=acc
            kernels['vxor'](one,zero,zero,bits.view(torch.uint32),out[start:start+64])
        return out[:n].reshape(x.shape)
    return truth,kernels,hashes


def logical(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    convert,kernels,hashes=_convert();name=case._call.signature.name
    op='vxor' if name=='LogicalNotFwdOp' else 'vand' if name=='LogicalAndFwdOp' else 'vor'
    def run(x,y=None):
        shape=tuple(x.shape) if y is None else tuple(torch.broadcast_shapes(x.shape,y.shape))
        a=convert(x).expand(shape).contiguous().flatten()
        b=torch.ones_like(a) if y is None else convert(y).expand(shape).contiguous().flatten()
        n=a.numel();a=F.pad(a,(0,-n%64));b=F.pad(b,(0,-n%64));out=torch.empty_like(a)
        zero=torch.zeros(64,device=x.device,dtype=torch.int32);mask=torch.ones_like(zero).view(torch.uint32)
        for start in range(0,a.numel(),64):kernels[op](a[start:start+64],b[start:start+64],zero,mask,out[start:start+64])
        return out[:n].to(torch.bool).reshape(shape)
    return Candidate(Implementation(run),'tilelang::merging_kernel native nonzero + '+op,
                     'Original unsigned mask comparison performs nonzero conversion. Original AND removes float sign bits so both signed zeros are false; original OR combines integer limbs and both complex components; FP64 bits are never narrowed. Native 0/1 bit operations perform logic; only final exact 0/1-to-bool cast is external.',hashes)


def extrema(case):
    import torch
    from benchmarks.api import Implementation
    from .reduce import extrema as original_extrema
    if case.inputs[0].dtype==torch.bool:return original_extrema(case)
    convert,kernels,hashes=_convert();x=case.inputs[0]
    # Shape/dtype-only placeholder; no input-derived values are cached.
    proxy=SimpleNamespace(inputs=(torch.empty(x.shape,device=x.device,dtype=torch.bool),),params=case.params,_call=case._call)
    candidate=original_extrema(proxy)
    def run(x):return candidate.implementation.run(convert(x).to(torch.bool))
    return replace(candidate,implementation=Implementation(run),source_sha256=candidate.source_sha256|hashes,
                   adaptation=candidate.adaptation+' Nonboolean input first passes through original native unsigned-mask nonzero conversion on every invocation.')


def count_nonzero(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from .microblocks import _load_builder
    convert,kernels,hashes=_convert();x=case.inputs[0]
    dims=case.params.get('dim');dims=tuple(range(x.ndim)) if dims is None else tuple(d%x.ndim for d in ((dims,) if isinstance(dims,int) else dims))
    outer=tuple(d for d in range(x.ndim) if d not in dims);order=outer+dims
    m,k=prod(x.shape[d] for d in outer),prod(x.shape[d] for d in dims)
    shape=tuple(case._call.tensors['output'][0])
    tl,builder,more=_load_builder('testing/ascend/language/test_tilelang_ascend_reduce.py','make_default_layout_reduce_kernel');hashes.update(more)
    kernel=tl.compile(builder(2,128,'int32','sum'),target='ascend',pass_configs={tl.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE:False})
    def run(x):
        data=convert(x).permute(order).contiguous().reshape(m,k)
        while True:
            packed=F.pad(data,(0,-data.shape[-1]%128)).reshape(-1,128);count=packed.shape[0]
            packed=F.pad(packed,(0,0,0,count%2));out=torch.empty(packed.shape[0],device=x.device,dtype=torch.int32)
            for start in range(0,packed.shape[0],2):kernel(packed[start:start+2],out[start:start+2])
            data=out[:count].reshape(m,-1)
            if data.shape[-1]==1:break
        return data.reshape(shape).to(torch.int64)
    return Candidate(Implementation(run),'tilelang::native unsigned-mask nonzero + int32 sum reducer',
                     'Original native masked comparison yields exact 0/1, original int32 sum reducer counts and combines partials. Manifest count extents fit int32; final widening to int64 is exact. No Torch nonzero/count/reduction.',hashes)
