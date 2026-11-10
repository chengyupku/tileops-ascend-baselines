"""Additional layouts for existing max and complete singleton-QK statistics."""
from math import prod
from .adapters import _prepare, _candidate, _reduce_rows, _bmm_single_key
from .contract import per_axis, pool_window


def logical_or(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p['output'][0])

    def run(x, other):
        import torch
        left = x.expand(shape).reshape(-1).to(torch.bfloat16)
        right = other.expand(shape).reshape(-1).to(torch.bfloat16)
        values = _reduce_rows(native, torch.stack((left, right), dim=1), 'min' if op in {'LogicalAndFwdOp', 'BitwiseAndFwdOp'} else 'max')
        return values.to(torch.bool).reshape(shape)

    return _candidate(run, hashes, 'Broadcast and losslessly widen bool0/1; pack paired lanes, select existing native max/min statistic and restore bool output.', op=op, params=p, statistic='max_logits.bool01')


def convolution(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p['input'][0])
    weight_shape = tuple(p['weight'][0])
    out_shape = tuple(p['output'][0])
    rank = len(shape) - 2
    kernel = weight_shape[2:]
    stride = per_axis(p.get('stride', 1), rank)
    dilation = per_axis(p.get('dilation', 1), rank)
    padding = per_axis(p.get('padding', 0), rank)
    groups = p.get('groups', 1)
    n, channels = shape[:2]
    cin = channels // groups
    cout = weight_shape[0] // groups
    spatial = out_shape[2:]
    width = prod(weight_shape[1:])
    pad = tuple(v for pad_value in reversed(padding) for v in (pad_value, pad_value))

    def run(x, weight, bias=None):
        import torch
        import torch.nn.functional as F
        source = F.pad(x, pad).reshape(n, groups, cin, *(s + 2 * d for s, d in zip(shape[2:], padding)))
        st = source.stride()
        windows = source.as_strided(
            (n, groups, cin, *spatial, *kernel),
            (*st[:3], *(s * t for s, t in zip(stride, st[3:])),
             *(d * t for d, t in zip(dilation, st[3:]))),
        )
        order = (0, 1, *range(3, 3 + rank), 2, *range(3 + rank, 3 + 2 * rank))
        a = windows.permute(order).contiguous().reshape(n * groups, prod(spatial), width)
        b = weight.reshape(groups, cout, width).transpose(1, 2)
        b = b[None].expand(n, -1, -1, -1).contiguous().reshape(n * groups, width, cout)
        if bias is not None:
            a = F.pad(a, (0, 1), value=1)
            bias_rows = bias.reshape(1, groups, 1, cout).expand(n, -1, -1, -1).reshape(n * groups, 1, cout)
            b = torch.cat((b, bias_rows), dim=1)
        values = _bmm_single_key(native, a, b)
        values = values.reshape(n, groups, *spatial, cout)
        return values.permute(0, 1, rank + 2, *range(2, rank + 2)).contiguous().reshape(out_shape)

    return _candidate(run, hashes, 'Expand exact convolution windows and weights into complete contractions of at most512 lanes. Optional bias uses a constant-one input lane. Reuse native singleton-QK maximum; copy/reshape the full N/C/spatial output without partial sums or kernel changes.', op=op, params=p, statistic='max_logits.single_key_QK.convolution')


def pool_scaled_sum(case):
    if case.params.get('divisor_override') is None:
        from .adapters import pool
        return pool(case)
    op, p, native, hashes = _prepare(case)
    spatial = tuple(p['input'][0][2:])
    output_shape = tuple(p['output'][0])
    outspatial = output_shape[2:]
    divisor = p['divisor_override']

    def run(x):
        import torch
        import torch.nn.functional as F
        from itertools import product
        flat = x.contiguous().reshape(-1, prod(spatial))
        output = torch.empty((flat.shape[0], prod(outspatial)), device=x.device, dtype=x.dtype)
        for linear, coord in enumerate(product(*(range(n) for n in outspatial))):
            locations, _, valid = pool_window(op, p, spatial, outspatial, coord)
            ids = torch.tensor(locations, device=x.device, dtype=torch.long)
            mask = torch.tensor(valid, device=x.device, dtype=torch.bool)
            rows = flat[:, ids].masked_fill(~mask[None], 0)
            count, width = rows.shape
            groups = (count + 63) // 64
            q = F.pad(rows, (0, 512 - width, 0, groups * 64 - count)).reshape(groups, 64, 512)
            kv = torch.zeros((1, 1, 512), device=x.device, dtype=x.dtype)
            kv[..., :width] = 1
            indices = torch.full((groups, 1, 64), -1, device=x.device, dtype=torch.int32)
            indices[..., 0] = 0
            _, scaled_sum, _ = native(q, kv, indices, 1.0 / divisor, 512, None, None)
            output[:, linear] = scaled_sum.reshape(-1)[:count].to(x.dtype)
        return output.reshape(output_shape)

    return _candidate(run, hashes, 'Gather complete pooling windows and pad with zero; original singleton QK against constant ones computes the full sum, and native final sm_scale=1/divisor applies the requested divisor. No partial sums or arithmetic outside FlashMLA.', op=op, params=p, statistic='max_logits.single_key_QK.scaled_window_sum')


def pointwise_dot(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p['output'][0])

    def run(x, other=None):
        import torch
        import torch.nn.functional as F
        left = x.expand(shape).reshape(-1)
        right = other.expand(shape).reshape(-1) if other is not None else None
        output = torch.empty_like(left)
        if x.dtype == torch.bool:
            left = left.to(torch.bfloat16)
            right = right.to(torch.bfloat16) if right is not None else None
        # Independent one-/two-lane complete contractions. The batch limit
        # bounds temporary storage; every original output element is retained.
        batch = 131072  # Q buffer is128MiB; original full contractions unchanged.
        for start in range(0, left.numel(), batch):
            stop = min(start + batch, left.numel())
            count = stop - start
            groups = (count + 63) // 64
            q = torch.zeros((groups, 64, 512), dtype=left.dtype, device=x.device)
            keys = 2 if op in {'AbsFwdOp', 'BitwiseXorFwdOp'} else 1
            kv = torch.zeros((groups, keys, 512), dtype=left.dtype, device=x.device)
            slots = torch.arange(64, device=x.device)
            a = F.pad(left[start:stop], (0, groups * 64 - count)).reshape(groups, 64)
            if op == 'MulFwdOp':
                b = F.pad(right[start:stop], (0, groups * 64 - count)).reshape(groups, 64)
                q[:, slots, slots] = a
                kv[:, 0, :64] = b
            elif op == 'AbsFwdOp':
                q[:, slots, slots] = 1
                q[:, slots, slots + 64] = -1
                kv[:, 0, :64] = a
                kv[:, 1, 64:128] = a
            elif op == 'BitwiseXorFwdOp':
                b = F.pad(right[start:stop], (0, groups * 64 - count)).reshape(groups, 64)
                q[:, slots, slots] = 1
                q[:, slots, slots + 64] = -1
                kv[:, 0, :64] = a
                kv[:, 0, 64:128] = b
                kv[:, 1, :64] = b
                kv[:, 1, 64:128] = a
            elif op == 'LogicalNotFwdOp':
                q[:, slots, slots] = -1
                q[:, :, 128] = 1
                kv[:, 0, :64] = a
                kv[:, 0, 128] = 1
            elif op == 'NegFwdOp':
                q[:, slots, slots] = -1
                kv[:, 0, :64] = a
            else:
                b = F.pad(right[start:stop], (0, groups * 64 - count)).reshape(groups, 64)
                q[:, slots, slots] = 1
                q[:, slots, slots + 64] = -1 if op == 'SubFwdOp' else 1
                kv[:, 0, :64] = a
                kv[:, 0, 64:128] = b
            indices = torch.full((groups, 1, 64), -1, device=x.device, dtype=torch.int32)
            indices[:, 0, :keys] = torch.arange(groups * keys, device=x.device, dtype=torch.int32).reshape(groups, keys)
            kv = kv.reshape(groups * keys, 1, 512)
            _, dot, _ = native(q, kv, indices, 1.0, 512, None, None)
            output[start:stop] = dot.reshape(-1)[:count].to(x.dtype)
        return output.reshape(shape)

    return _candidate(run, hashes, 'Pack broadcast operands into independent complete one-/two-key QK projections with exact unit constant lanes. Select native max_logits, cast to the original BF16 or exact bool0/1 output and restore its full shape. No partial sums, input-dependent host computation or external numerical operations.', op=op, params=p, statistic='max_logits.single_key_QK.pointwise')


def softplus(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p['output'][0])

    def run(x):
        import torch
        values = x.reshape(-1)
        rows = torch.stack((values, torch.zeros_like(values)), dim=1)
        return _reduce_rows(native, rows, 'lse').reshape(shape)

    return _candidate(run, hashes, 'Pack each input with constant zero and select the existing natural-log LSE statistic for original beta1/threshold20 BF16 softplus. Full outputs and original tolerance are checked; no input-dependent cache or outside floating-point arithmetic.', op=op, params=p, statistic='lse.input_and_zero')


def vector_infinity_norm(case):
    op, p, native, hashes = _prepare(case)
    from .contract import axes_plan
    remaining, axes, width, shape = axes_plan(p['x'][0], p.get('dim'), p.get('keepdim', False))

    def run(x):
        import torch
        import torch.nn.functional as F
        rows = x.permute(*remaining, *axes).contiguous().reshape(-1, width)
        output = torch.empty((rows.shape[0],), dtype=getattr(torch, p['output'][1]), device=x.device)
        groups_per_batch = max(1, min(32, (32 * 1024**2) // (2 * width * 512 * 2)))
        for start in range(0, rows.shape[0], 64 * groups_per_batch):
            stop = min(start + 64 * groups_per_batch, rows.shape[0])
            count = stop - start
            groups = (count + 63) // 64
            packed = F.pad(rows[start:stop], (0, 0, 0, groups * 64 - count)).reshape(groups, 64, width).transpose(1, 2)
            kv = torch.zeros((groups, 2 * width, 1, 512), device=x.device, dtype=x.dtype)
            kv[:, :width, 0, :64] = packed
            kv[:, width:, 0, 64:128] = packed
            kv = kv.reshape(groups * 2 * width, 1, 512)
            q = torch.zeros((groups, 64, 512), device=x.device, dtype=x.dtype)
            diagonal = torch.arange(64, device=x.device)
            q[:, diagonal, diagonal] = 1
            q[:, diagonal, diagonal + 64] = -1
            indices = torch.arange(groups * 2 * width, device=x.device, dtype=torch.int32).reshape(groups, 1, 2 * width)
            indices = F.pad(indices, (0, (-2 * width) % 64), value=-1)
            _, maximum, _ = native(q, kv, indices, 1.0, 512, None, None)
            output[start:stop] = maximum.reshape(-1)[:count].to(output.dtype)
        return output.reshape(shape)

    return _candidate(run, hashes, 'Retain every original reduction-axis value in positive and negative constant-basis key slots. Original native maximum over both complete key domains directly gives infinity norm; reshape the original output and preserve requested BF16/FP32 dtype. No external absolute value or numerical reduction.', op=op, params=p, statistic='max_logits.complete_signed_basis_domain')


def two_key_activation(case):
    op, p, native, hashes = _prepare(case)
    shape = tuple(p['output'][0])

    def run(x):
        import torch
        import torch.nn.functional as F
        flat = x.reshape(-1)
        output = torch.empty_like(flat)
        for start in range(0, flat.numel(), 131072):
            stop = min(start + 131072, flat.numel())
            count = stop - start
            groups = (count + 63) // 64
            values = F.pad(flat[start:stop], (0, groups * 64 - count)).reshape(groups, 64)
            q = torch.zeros((groups, 64, 512), device=x.device, dtype=x.dtype)
            slots = torch.arange(64, device=x.device)
            q[:, slots, slots] = 2 if op == 'TanhFwdOp' else 1
            kv = torch.zeros((groups, 2, 512), device=x.device, dtype=x.dtype)
            kv[:, 0, :64] = values
            if op == 'SiluFwdOp':
                kv[:, 0, 64:128] = values
            else:
                kv[:, 0, 64] = 1
                if op == 'TanhFwdOp':
                    kv[:, 1, 64] = -1
            indices = torch.full((groups, 1, 64), -1, device=x.device, dtype=torch.int32)
            indices[:, 0, :2] = torch.arange(groups * 2, device=x.device, dtype=torch.int32).reshape(groups, 2)
            result, _, _ = native(q, kv.reshape(groups * 2, 1, 512), indices, 1.0, 512, None, None)
            selected = result[:, slots, slots + 64] if op == 'SiluFwdOp' else result[:, :, 64]
            output[start:stop] = selected.reshape(-1)[:count]
        return output.reshape(shape)

    return _candidate(run, hashes, 'Original two-key attention directly returns sigmoid from scores[x,0] and values[1,0], SiLU from scores[x,0] and values[x,0], or tanh from scores[2x,0] and values[1,-1]. Constant query lanes and value copies are layouts only; all exponentiation, normalization and weighted sums remain in the unchanged native API.', op=op, params=p, statistic='output.complete_two_key_activation')


def softmax_basis(case):
    op, p, native, hashes = _prepare(case)
    from .contract import axes_plan
    remaining, axes, width, shape = axes_plan(p['x'][0], p.get('dim'), False)
    order = (*remaining, *axes)
    inverse = tuple(order.index(i) for i in range(len(order)))

    def run(x):
        import torch
        import torch.nn.functional as F
        rows = x.permute(order).contiguous().reshape(-1, width)
        output = torch.empty_like(rows)
        groups_per_batch = max(1, min(32, (128 * 1024**2) // (width * 512 * 2)))
        slots = torch.arange(64, device=x.device)
        for start in range(0, rows.shape[0], 64 * groups_per_batch):
            stop = min(start + 64 * groups_per_batch, rows.shape[0])
            count = stop - start
            groups = (count + 63) // 64
            packed = F.pad(rows[start:stop], (0, 0, 0, groups * 64 - count)).reshape(groups, 64, width).transpose(1, 2)
            q = torch.zeros((groups, 64, 512), device=x.device, dtype=x.dtype)
            q[:, slots, slots] = 1
            kv = torch.zeros((groups, width, 512), device=x.device, dtype=x.dtype)
            kv[:, :, :64] = packed
            indices = torch.arange(groups * width, device=x.device, dtype=torch.int32).reshape(groups, 1, width)
            indices = F.pad(indices, (0, (-width) % 64), value=-1)
            for col in range(0, width, 448):
                end = min(col + 448, width)
                kv[:, :, 64:] = 0
                local = torch.arange(end - col, device=x.device)
                kv[:, col + local, 64 + local] = 1
                result, _, _ = native(q, kv.reshape(groups * width, 1, 512), indices, 1.0, 512, None, None)
                output[start:stop, col:end] = result[:, :, 64:64 + end - col].reshape(groups * 64, end - col)[:count]
        return output.reshape(*(x.shape[i] for i in order)).permute(inverse).contiguous()

    return _candidate(run, hashes, 'Copy each complete original score domain into independent basis-query lanes. Constant one-hot value columns expose softmax weights directly through native weighted output. Split only output columns into448-wide projections; every projection retains all original keys and original normalization. No probability arithmetic, partial denominator or truncated domain is used.', op=op, params=p, statistic='output.complete_domain_identity_value_projection')
