"""Layout-only compositions. Arithmetic is supplied exclusively by the owned engine.

This file is duplicated in CATLASS/TLA to keep provider ownership independent.
All input reads, casts, constants, packing, copies and launches occur in run.
No partial reduction, split GEMM K, native body, tile or schedule is introduced.
"""
from itertools import product
from math import prod

SCRATCH_ELEMENTS = 262144


def axes_layout(shape, dim):
    axes = list(range(len(shape))) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    if any(not -len(shape) <= a < len(shape) for a in axes):
        raise ValueError('Reduction dimension out of range')
    axes = [a % len(shape) for a in axes]
    if len(set(axes)) != len(axes) or not axes:
        raise ValueError('Expected nonempty unique reduction dimensions')
    axes.sort()
    rest = [i for i in range(len(shape)) if i not in axes]
    return rest + axes, prod(shape[i] for i in rest), prod(shape[i] for i in axes)


def pieces(shape, budget=SCRATCH_ELEMENTS):
    """Bounded rectangular slices without materializing a broadcasted full input."""
    if prod(shape) <= budget:
        yield (slice(None),) * len(shape)
        return
    tail = prod(shape[1:])
    if tail <= budget:
        step = max(1, budget // tail)
        for i in range(0, shape[0], step):
            yield (slice(i, min(i + step, shape[0])),) + (slice(None),) * (len(shape)-1)
    else:
        for i in range(shape[0]):
            for sub in pieces(shape[1:], budget):
                yield (slice(i, i+1),) + sub


def tuple_param(value, rank):
    result = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(result) != rank:
        raise ValueError('Invalid spatial parameter rank')
    return result


def pool_geometry(p):
    shape = tuple(p['input'][0]); rank = len(p['output'][0]) - 2
    spatial = shape[-rank:]
    k = tuple_param(p['kernel_size'], rank)
    s = tuple_param(p.get('stride') if p.get('stride') is not None else k, rank)
    pad = tuple_param(p.get('padding', 0), rank)
    out = []
    for n, size, step, padding in zip(spatial, k, s, pad):
        extent = (n + 2*padding - size + (step-1 if p.get('ceil_mode') else 0)) // step + 1
        if p.get('ceil_mode') and (extent-1)*step >= n+padding:
            extent -= 1
        if extent <= 0 or size <= 0 or step <= 0:
            raise ValueError('Invalid pooling geometry')
        out.append(extent)
    return spatial, k, s, pad, tuple(out)


def pool_window(spatial, k, s, pad, index, p):
    begin = tuple(i*step-padding for i,step,padding in zip(index,s,pad))
    lo = tuple(max(0,a) for a in begin)
    hi = tuple(min(n,a+size) for n,a,size in zip(spatial,begin,k))
    if p.get('divisor_override') is not None:
        divisor = p['divisor_override']
    elif p.get('count_include_pad', True):
        divisor = prod(max(0,min(a+size,n+padding)-max(a,-padding))
                       for a,size,n,padding in zip(begin,k,spatial,pad))
    else:
        divisor = prod(max(0,b-a) for a,b in zip(lo,hi))
    if divisor == 0:
        raise ValueError('Empty pooling divisor')
    return begin, lo, hi, divisor


def constant(value, like):
    import torch
    return torch.full((), value, dtype=torch.float32, device=like.device)


def mean_rows(engine, matrix, divisor):
    summed = engine.reduce(matrix, 'sum')
    return engine.binary('div', summed, constant(divisor, summed))


def build_run(op, case, engine):
    """Return the complete manifest call; no input data is cached here."""
    import torch
    p = case.params
    key = 'x' if op.startswith('INT8Dequant') else 'cb' if op == 'SSDChunkCouplingFwdOp' else 'output'
    out_shape, dtype_name = p[key]
    out_shape, out_dtype = tuple(out_shape), getattr(torch, dtype_name)

    if op in {'RMSNormFwdOp','FusedAddRMSNormFwdOp','FusedAddLayerNormFwdOp',
              'AdaLayerNormFwdOp','AdaLayerNormZeroFwdOp'}:
        shape = tuple(p['x'][0])
        hidden = prod(p.get('normalized_shape', (shape[-1],)))
        rows = prod(shape)//hidden
        eps = 2**-23 if p.get('eps') is None else p['eps']
        fused = op.startswith('FusedAdd')
        adaptive = op.startswith('Ada')
        rms = 'RMSNorm' in op
        step = max(1,SCRATCH_ELEMENTS//hidden)
        def run(x, *args):
            output = torch.empty(shape, dtype=out_dtype, device=x.device)
            residual_output = torch.empty(shape, dtype=x.dtype, device=x.device) if fused else None
            flat = x.reshape(rows,hidden)
            for start in range(0,rows,step):
                stop = min(rows,start+step)
                value = flat[start:stop].float().contiguous()
                if fused:
                    residual = args[0].reshape(rows,hidden)[start:stop].float().contiguous()
                    rounded = engine.binary('add',value,residual).to(x.dtype)
                    residual_output.reshape(rows,hidden)[start:stop].copy_(rounded)
                    value = rounded.float()
                if rms:
                    y = engine.rms(value,eps)
                else:
                    # Shift before summing as well as before squaring. The original
                    # row-reduce accumulates FP32 block sums serially; summing a
                    # large common offset can spoil even a constant row's mean.
                    shifted = engine.binary('sub',value,value[:,:1])
                    mu = mean_rows(engine,shifted,hidden).reshape(-1,1)
                    centered = engine.binary('sub',shifted,mu)
                    variance = mean_rows(engine,engine.binary('mul',centered,centered),hidden)
                    std = engine.unary('sqrt',engine.binary('add',variance,constant(eps,variance)))
                    y = engine.binary('div',centered,std.reshape(-1,1))
                if adaptive:
                    scale,shift,*gate = args
                    scale = scale.expand(shape).reshape(rows,hidden)[start:stop].float()
                    shift = shift.expand(shape).reshape(rows,hidden)[start:stop].float()
                    y = engine.binary('add',engine.binary('mul',y,scale),shift)
                    if gate:
                        g = gate[0].expand(shape).reshape(rows,hidden)[start:stop].float()
                        y = engine.binary('mul',y,g)
                else:
                    affine = args[1:] if fused else args
                    weight = affine[0] if affine else None
                    bias = affine[1] if len(affine)>1 else None
                    if weight is not None:
                        y = engine.binary('mul',y,weight.reshape(1,hidden).float())
                    if bias is not None:
                        y = engine.binary('add',y,bias.reshape(1,hidden).float())
                output.reshape(rows,hidden)[start:stop].copy_(y.to(out_dtype))
            return (output,residual_output) if fused else output
        return run

    if op in {'MeanFwdOp','VectorNormFwdOp','AnyFwdOp','AllFwdOp','CountNonzeroFwdOp'}:
        shape = tuple(p['x'][0]); perm,rows,width = axes_layout(shape,p.get('dim'))
        step = max(1,SCRATCH_ELEMENTS//width)
        order = p.get('ord',2)
        def run(x):
            matrix = x.permute(perm).contiguous().reshape(rows,width)
            output = torch.empty(rows,dtype=out_dtype,device=x.device)
            for start in range(0,rows,step):
                v = matrix[start:start+step].float()
                if op == 'MeanFwdOp':
                    y = mean_rows(engine,v,width)
                elif op == 'VectorNormFwdOp':
                    v = engine.binary('mul',v,v) if order == 2 else engine.unary('abs',v)
                    y = engine.reduce(v,'max' if order in ('inf',float('inf')) else 'sum')
                    if order == 2: y = engine.unary('sqrt',y)
                else:
                    truth = v if x.dtype == torch.bool else engine.truth(v)
                    y = engine.reduce(truth.float(),{'AnyFwdOp':'max','AllFwdOp':'min','CountNonzeroFwdOp':'sum'}[op])
                output[start:start+step].copy_(y.to(out_dtype))
            return output.reshape(out_shape)
        return run

    if op == 'MeanPoolingFwdOp':
        chunk = p['chunk_size']
        def run(x,offsets=None,indices=None):
            if offsets is None:
                if indices is not None: raise ValueError('indices requires offsets')
                intervals = [(i,min(i+chunk,x.shape[1])) for i in range(0,x.shape[1],chunk)]
            else:
                if indices is None: raise ValueError('offsets requires indices')
                bounds = offsets.tolist(); pairs = indices.tolist()
                if len(bounds)<2 or bounds[0]!=0 or bounds[-1]!=x.shape[1] or any(a>b for a,b in zip(bounds,bounds[1:])):
                    raise ValueError('Invalid current sequence offsets')
                intervals=[]
                for seq,part in pairs:
                    if not 0<=seq<len(bounds)-1 or part<0: raise ValueError('Invalid current chunk index')
                    start=bounds[seq]+part*chunk; stop=min(start+chunk,bounds[seq+1])
                    if start>=stop: raise ValueError('Empty current chunk')
                    intervals.append((start,stop))
            expected=(x.shape[0],len(intervals),*x.shape[2:])
            if expected != out_shape: raise ValueError('Current chunk count disagrees with output contract')
            output=torch.empty(out_shape,dtype=out_dtype,device=x.device)
            for index,(lo,hi) in enumerate(intervals):
                matrix=x[:,lo:hi].permute(0,2,3,1).contiguous().reshape(-1,hi-lo)
                step=max(1,SCRATCH_ELEMENTS//(hi-lo))
                values=torch.empty(matrix.shape[0],dtype=out_dtype,device=x.device)
                for row in range(0,matrix.shape[0],step):
                    values[row:row+step].copy_(mean_rows(engine,matrix[row:row+step].float(),hi-lo).to(out_dtype))
                output[:,index].copy_(values.reshape(x.shape[0],*x.shape[2:]))
            return output
        return run

    if op.startswith('AvgPool'):
        spatial,k,stride,pad,spatial_out=pool_geometry(p)
        rank=len(k); leading=tuple(p['input'][0][:-rank]); planes=prod(leading)
        step=max(1,SCRATCH_ELEMENTS//prod(k))
        def run(x):
            output=torch.empty(out_shape,dtype=out_dtype,device=x.device)
            source=x.reshape(planes,*spatial); target=output.reshape(planes,*spatial_out)
            for index in product(*(range(n) for n in spatial_out)):
                begin,lo,hi,divisor=pool_window(spatial,k,stride,pad,index,p)
                src_slice=tuple(slice(a,b) for a,b in zip(lo,hi))
                dst_slice=tuple(slice(a-b,c-b) for a,c,b in zip(lo,hi,begin))
                for row in range(0,planes,step):
                    count=min(step,planes-row)
                    packed=torch.zeros((count,*k),dtype=torch.float32,device=x.device)
                    packed[(slice(None),*dst_slice)].copy_(source[(slice(row,row+count),*src_slice)])
                    y=mean_rows(engine,packed.reshape(count,-1),divisor)
                    target[(slice(row,row+count),*index)].copy_(y.to(out_dtype))
            return output
        return run

    if op == 'AdaptiveAvgPool2dFwdOp':
        spatial=tuple(p['input'][0][-2:]); size=tuple(p['output_size']); leading=tuple(p['input'][0][:-2])
        def run(x):
            output=torch.empty(out_shape,dtype=out_dtype,device=x.device)
            for index in product(*(range(n) for n in size)):
                lo=tuple(i*n//o for i,n,o in zip(index,spatial,size))
                hi=tuple(((i+1)*n+o-1)//o for i,n,o in zip(index,spatial,size))
                width=prod(b-a for a,b in zip(lo,hi))
                matrix=x[(...,*(slice(a,b) for a,b in zip(lo,hi)))].contiguous().reshape(-1,width)
                values=torch.empty(matrix.shape[0],dtype=out_dtype,device=x.device)
                step=max(1,SCRATCH_ELEMENTS//width)
                for row in range(0,matrix.shape[0],step):
                    values[row:row+step].copy_(mean_rows(engine,matrix[row:row+step].float(),width).to(out_dtype))
                output[(...,*index)].copy_(values.reshape(leading))
            return output
        return run

    if op == 'SSDChunkCouplingFwdOp':
        b,s,g,d=p['C_mat'][0]; q=p['chunk_len']
        def run(c,bmat):
            output=torch.empty(out_shape,dtype=out_dtype,device=c.device)
            mask=torch.zeros((q,q),dtype=torch.float32,device=c.device)
            for row in range(q):
                mask[row,:row+1].fill_(1)
            for batch,chunk,group in product(range(b),range(s//q),range(g)):
                lo=chunk*q; hi=lo+q
                # The oracle masks the FP32 dot product before its final cast.
                # Masking a half result would turn finite overflow into NaN;
                # copying only the lower triangle would suppress genuine NaNs.
                full=engine.gemm(c[batch,lo:hi,group].float().contiguous(),
                                 bmat[batch,lo:hi,group].float().T.contiguous())
                output[batch,chunk,group].copy_(engine.binary('mul',full,mask).to(out_dtype))
            return output
        return run

    if op.startswith('INT8Dequant'):
        rows,width=out_shape; step=max(1,SCRATCH_ELEMENTS//width)
        def run(q,scale):
            output=torch.empty(out_shape,dtype=out_dtype,device=q.device)
            for row in range(0,rows,step):
                value=q[row:row+step].float()
                if op=='INT8DequantPerBlockFwdOp':
                    expanded=scale[row:row+step].repeat_interleave(128,-1)[...,:width]
                elif op=='INT8DequantPerChannelFwdOp': expanded=scale[row:row+step].reshape(-1,1)
                else: expanded=scale
                output[row:row+step].copy_(engine.binary('mul',value,expanded.float()).to(out_dtype))
            return output
        return run

    if op in {'ClampTensorFwdOp','ClampScalarFwdOp','HardtanhFwdOp','RsqrtFwdOp','LogicalOrFwdOp','LogicalAndFwdOp'}:
        def run(x,*args):
            output=torch.empty(out_shape,dtype=out_dtype,device=x.device)
            expanded=x.expand(out_shape)
            for sl in pieces(out_shape):
                value=expanded[sl].float().contiguous()
                if op.startswith('Logical'):
                    other=args[0].expand(out_shape)[sl].float().contiguous()
                    y=engine.logical('or' if op=='LogicalOrFwdOp' else 'and',engine.truth(value),engine.truth(other))
                elif op=='RsqrtFwdOp':
                    root=engine.unary('sqrt',value)
                    y=engine.binary('div',constant(1,root),root)
                else:
                    if op=='ClampTensorFwdOp':
                        # Manifest factories pass explicit None for absent optional bounds.
                        lo=args[0] if args else None; hi=args[1] if len(args)>1 else None
                        lo=lo.expand(out_shape)[sl].float() if lo is not None else None
                        hi=hi.expand(out_shape)[sl].float() if hi is not None else None
                    else:
                        lower=p.get('min_val',-1.) if op=='HardtanhFwdOp' else p.get('min')
                        upper=p.get('max_val',1.) if op=='HardtanhFwdOp' else p.get('max')
                        lo=constant(lower,value) if lower is not None else None
                        hi=constant(upper,value) if upper is not None else None
                    y=engine.binary('max',value,lo) if lo is not None else value
                    if hi is not None: y=engine.binary('min',y,hi)
                output[sl].copy_(y.to(out_dtype))
            return output
        return run
    raise ValueError(op)
