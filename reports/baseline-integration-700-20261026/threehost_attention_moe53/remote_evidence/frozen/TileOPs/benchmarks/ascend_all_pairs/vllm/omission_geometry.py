"""Static shape/index plans for the omission proposal; never reads tensor data."""
from itertools import product
from math import prod


def vector(value, rank):
    result = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(result) != rank:
        raise ValueError('Spatial parameter rank mismatch')
    return result


def coordinates(start, stop, shape):
    """Python integer coordinates only, in canonical contiguous order."""
    divisors = [prod(shape[i + 1:]) for i in range(len(shape))]
    return [[(j // step) % size for j in range(start, stop)]
            for step, size in zip(divisors, shape)]


def convolution_geometry(p):
    shape, dtype = p['input']
    weight, wtype = p['weight']
    rank = len(shape) - 2
    if rank not in (1, 2, 3) or len(weight) != rank + 2:
        raise ValueError('Expected batched 1D/2D/3D convolution')
    stride = vector(p.get('stride', 1), rank)
    dilation = vector(p.get('dilation', 1), rank)
    kernel = tuple(weight[2:])
    padding = p.get('padding', 0)
    if padding == 'valid':
        pads = ((0, 0),) * rank
    elif padding == 'same':
        if any(s != 1 for s in stride):
            raise ValueError('Original convolution same padding requires unit stride')
        total = [d * (k - 1) for d, k in zip(dilation, kernel)]
        pads = tuple((v // 2, v - v // 2) for v in total)
    elif isinstance(padding, str):
        raise ValueError('Unknown padding')
    else:
        pads = tuple((v, v) for v in vector(padding, rank))
    groups = p.get('groups', 1)
    if (groups < 1 or shape[1] % groups or weight[0] % groups
            or weight[1] != shape[1] // groups):
        raise ValueError('Inconsistent convolution groups/channels')
    if dtype != wtype or dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('Convolution requires matching floating operands')
    if 'bias' in p and (tuple(p['bias'][0]) != (weight[0],) or p['bias'][1] != dtype):
        raise ValueError('Bias shape/dtype mismatch')
    if any(s <= 0 for s in stride + dilation + kernel) or any(a < 0 for t in pads for a in t):
        raise ValueError('Invalid convolution geometry')
    spatial = tuple((n + lo + hi - d * (k - 1) - 1) // s + 1
                    for n, (lo, hi), d, k, s in zip(shape[2:], pads, dilation, kernel, stride))
    expected = (shape[0], weight[0], *spatial)
    if any(v <= 0 for v in spatial) or tuple(p['output'][0]) != expected or p['output'][1] != dtype:
        raise ValueError('Declared convolution output does not match geometry')
    return dict(rank=rank, stride=stride, dilation=dilation, kernel=kernel,
                pads=pads, groups=groups, spatial=spatial, cin=weight[1],
                cout=weight[0] // groups, k=prod(weight[1:]))


def coupling_geometry(p):
    shape, dtype = p['C_mat']
    if len(shape) != 4 or p['B_mat'] != p['C_mat']:
        raise ValueError('Coupling requires matching [B,S,G,D] operands')
    batch, length, groups, state = shape
    q = p['chunk_len']
    if q <= 0 or length % q:
        raise ValueError('Coupling reference requires complete chunks (no ragged tail)')
    expected = (batch, length // q, groups, q, q)
    if tuple(p['cb'][0]) != expected or p['cb'][1] != dtype:
        raise ValueError('Coupling output mismatch')
    if dtype not in ('float16', 'bfloat16', 'float32'):
        raise ValueError('No exact floating contraction for this dtype')
    return batch, length // q, groups, q, state


def reduction_geometry(p):
    shape, _ = p['x']
    rank = len(shape)
    dims = p.get('dim')
    dims = tuple(range(rank)) if dims is None or dims == [] or dims == () else ((dims,) if isinstance(dims, int) else tuple(dims))
    if any(d < -rank or d >= rank for d in dims):
        raise ValueError('Reduction axis out of range')
    dims = tuple(d % rank for d in dims)
    if len(set(dims)) != len(dims):
        raise ValueError('Repeated reduction axis')
    order = tuple(d for d in range(rank) if d not in dims) + dims
    width = prod(shape[d] for d in dims)
    expected = tuple(1 if i in dims else n for i, n in enumerate(shape)) if p.get('keepdim', False) else tuple(n for i, n in enumerate(shape) if i not in dims)
    if tuple(p['output'][0]) != expected or width <= 0:
        raise ValueError('Reduction output mismatch or empty extrema domain')
    return order, width


def pool_geometry(p, adaptive=False):
    shape, dtype = p['input']
    output, otype = p['output']
    rank = 2 if adaptive else (1 if isinstance(p['kernel_size'], int) and len(shape) == 3 else len(shape) - 2)
    # The non-adaptive proposals are batched; adaptive supports CHW and NCHW.
    spatial = tuple(shape[-rank:])
    out = tuple(output[-rank:])
    if dtype != otype or tuple(shape[:-rank]) != tuple(output[:-rank]):
        raise ValueError('Pool output prefix/dtype mismatch')
    if adaptive:
        size = p.get('output_size')
        size = (None, None) if size is None else vector(size, 2)
        expected = tuple(n if s is None else s for n, s in zip(spatial, size))
        if expected != out or min(out) <= 0:
            raise ValueError('Adaptive pool output mismatch')
        return dict(rank=rank, spatial=spatial, out=out, adaptive=True)
    kernel = vector(p['kernel_size'], rank)
    stride = kernel if p.get('stride') is None else vector(p['stride'], rank)
    padding = vector(p.get('padding', 0), rank)
    dilation = vector(p.get('dilation', 1), rank)
    expected = []
    if any(v <= 0 for v in kernel + stride + dilation) or any(v < 0 for v in padding):
        raise ValueError('Invalid maxpool geometry')
    for n, k, s, pad, d in zip(spatial, kernel, stride, padding, dilation):
        if pad > k // 2:
            raise ValueError('Pool padding exceeds original API bound')
        size = (n + 2 * pad - d * (k - 1) - 1 + (s - 1 if p.get('ceil_mode') else 0)) // s + 1
        if p.get('ceil_mode') and (size - 1) * s >= n + pad:
            size -= 1
        expected.append(size)
    if tuple(expected) != out or min(out) <= 0:
        raise ValueError('Pool output mismatch')
    return dict(rank=rank, spatial=spatial, out=out, adaptive=False,
                kernel=kernel, stride=stride, padding=padding, dilation=dilation)


def pool_window(plan, output_flat):
    """Complete valid spatial window; omit padding instead of adding -inf data."""
    pos = [a[0] for a in coordinates(output_flat, output_flat + 1, plan['out'])]
    if plan['adaptive']:
        axes = [range(i * n // o, ((i + 1) * n + o - 1) // o)
                for i, n, o in zip(pos, plan['spatial'], plan['out'])]
    else:
        axes = [[i * s - pad + j * d for j in range(k)
                 if 0 <= i * s - pad + j * d < n]
                for i, s, pad, k, d, n in zip(pos, plan['stride'], plan['padding'],
                                             plan['kernel'], plan['dilation'], plan['spatial'])]
    multipliers = [prod(plan['spatial'][i + 1:]) for i in range(plan['rank'])]
    return tuple(sum(i * m for i, m in zip(loc, multipliers)) for loc in product(*axes))
