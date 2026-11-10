"""Complete candidates for URA-001..004/035..046 (not native-verified).

Only layout/copy/static construction and exact necessary casts use Torch.
All contractions, value extrema and sign/mask arithmetic call original APIs.
Native bodies/decorators/launch schedules are never extracted or modified.
"""
from math import prod
from pathlib import Path

from .omission_geometry import (convolution_geometry, coordinates, coupling_geometry,
                                pool_geometry, pool_window, reduction_geometry)
from .omission_contracts import omission_decision
from .runtime import TRITON_ROOT, candidate, module, validate

# Bound only independent output rows, never the mathematical reduction domain.
PACK_BYTES = 64 * 1024 * 1024
MAX_GEMM_ROWS = 4096
MAX_STAT_ROWS = 8192


def _sources():
    here = Path(__file__)
    return [here, here.with_name('omission_geometry.py'),
            here.with_name('omission_contracts.py'), here.with_name('omission_admission.json')]


def _indices(start, stop, shape, device):
    import torch
    return tuple(torch.tensor(a, dtype=torch.long, device=device)
                 for a in coordinates(start, stop, shape))


def _flat_slice(x, shape, begin, end):
    expanded = x.expand(shape)
    if expanded.is_contiguous():
        return expanded.reshape(-1)[begin:end]
    return expanded[_indices(begin, end, shape, x.device)]


def _convolution_run(p, matmul, *, max_rows=MAX_GEMM_ROWS, pack_bytes=PACK_BYTES):
    """Testable exact production packing; matmul is the original public API."""
    plan = convolution_geometry(p)

    def run(x, weight, bias=None):
        import torch
        rank, pads = plan['rank'], plan['pads']
        # The entire current input and weight are read again on EVERY invocation.
        data = x.new_zeros((x.shape[0], x.shape[1],
                            *(n + lo + hi for n, (lo, hi) in zip(x.shape[2:], pads))))
        data[(slice(None), slice(None), *(slice(lo, lo + n) for n, (lo, hi) in zip(x.shape[2:], pads)))].copy_(x)
        spatial = plan['spatial']
        count, oc = prod(spatial), weight.shape[0]
        result = torch.empty((x.shape[0], count, oc), dtype=x.dtype, device=x.device)
        rows = max(1, min(max_rows, pack_bytes // (x.element_size() * (plan['k'] + plan['cout']))))
        st = data.stride()
        view_size = (*spatial, plan['cin'], *plan['kernel'])
        view_stride = (*(s * d for s, d in zip(st[2:], plan['stride'])), st[1],
                       *(s * d for s, d in zip(st[2:], plan['dilation'])))
        for group in range(plan['groups']):
            clo, olo = group * plan['cin'], group * plan['cout']
            packed_weight = weight[olo:olo + plan['cout']].reshape(plan['cout'], plan['k']).transpose(0, 1).contiguous()
            packed_bias = None if bias is None else bias[olo:olo + plan['cout']].contiguous()
            for batch in range(x.shape[0]):
                patches = data.as_strided(view_size, view_stride,
                                          batch * st[0] + clo * st[1])
                for begin in range(0, count, rows):
                    end = min(begin + rows, count)
                    # Index only independent spatial rows BEFORE materializing im2col.
                    packed = patches[_indices(begin, end, spatial, x.device)].contiguous().reshape(end - begin, plan['k'])
                    out = matmul(packed, packed_weight, bias=packed_bias)
                    result[batch, begin:end, olo:olo + plan['cout']].copy_(out)
        return result.reshape(x.shape[0], *spatial, oc).movedim(-1, 1).contiguous()

    return run


def convolution(case):
    # Preserve the existing pointwise provider exactly; this proposal owns ONLY
    # the newly enumerated non-unit-window cases, not the simultaneous review.
    if omission_decision(case._call.signature.name, case.params) is None:
        from .basic import pointwise_conv
        return pointwise_conv(case)
    validate(case)
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    return candidate(_convolution_run(case.params, native.matmul_persistent),
        'matmul_persistent (complete im2col convolution)',
        'Per-call zero padding and complete channel/kernel im2col; groups and output-row slabs only. '
        'Every GEMM retains the entire K, original FP32 accumulation, bias epilogue and fixed 128x128x64 schedule. '
        'Restore every NCHW/NCDHW output; all copies/packing and native calls are measured.',
        [*_sources(), *sources], timing_mode='host_completed')


def _coupling_run(p, matmul, muls_add):
    batch, chunks, groups, q, state = coupling_geometry(p)

    def run(c_mat, b_mat):
        import torch
        # Exact widening is necessary: the ORIGINAL oracle masks in FP32 before
        # casting. Half overflow before 0*upper would otherwise create false NaNs.
        c = c_mat.float().reshape(batch, chunks, q, groups, state)
        b = b_mat.float().reshape(batch, chunks, q, groups, state)
        result = torch.empty(tuple(p['cb'][0]), dtype=c_mat.dtype, device=c_mat.device)
        for bi in range(batch):
            for ci in range(chunks):
                for gi in range(groups):
                    full = matmul(c[bi, ci, :, gi].contiguous(),
                                  b[bi, ci, :, gi].transpose(0, 1).contiguous())
                    flat = full.reshape(-1, 1)
                    # This preserves the current oracle's IEEE upper-triangle
                    # semantics: finite*0 -> signed zero, Inf/NaN*0 -> NaN.
                    # -0 addend preserves sign of a zero product under IEEE add.
                    masked = muls_add(flat, torch.full_like(flat, -0.0), 0.0).reshape(q, q)
                    for row in range(q):
                        masked[row, :row + 1].copy_(full[row, :row + 1])
                    result[bi, ci, gi].copy_(masked.to(c_mat.dtype))
        return result

    return run


def ssd_coupling(case):
    validate(case)
    native, sources = module(TRITON_ROOT + 'batch_invariant/matmul.py')
    arithmetic, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
    return candidate(_coupling_run(case.params, native.matmul_persistent, arithmetic.muls_add_triton),
        'matmul_persistent + muls_add_triton (causal SSD coupling)',
        'Exact FP32 widening; whole C_chunk @ B_chunk.T for every batch/chunk/group. '
        'Original scalar multiply-add supplies FP32 zero masking, then copy the causal lower triangle '
        'and cast once, matching current oracle including nonfinite upper entries. All outputs/packing timed.',
        [*_sources(), *sources, *extra], timing_mode='host_completed', limitations=(
            'The audit suggested literal zero initialization, but current oracle multiplies by a mask: '
            'nonfinite upper-triangle products must remain NaN. This candidate follows the unchanged oracle.',))


def _stat_row_limit(width, *, pack_bytes=PACK_BYTES, max_rows=MAX_STAT_ROWS):
    # K and RoPE plus Q/QR/output/indices/max/sum; values alias K.
    # Library-private workspace is separate and depends on its original tiler.
    bytes_per_row = width * (576 * 2 + 4) + 576 * 2 + 512 * 2 + 8
    return max(1, min(max_rows, pack_bytes // bytes_per_row))


def _stat_max(data, ops, *, pack_bytes=PACK_BYTES, max_rows=MAX_STAT_ROWS):
    """Use the unchanged sparse-attention statistic on each WHOLE input row."""
    import torch
    rows, width = data.shape
    if width <= 0 or data.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Sparse max requires nonempty FP16/BF16 rows')
    out = torch.empty(rows, dtype=data.dtype, device=data.device)
    limit = _stat_row_limit(width, pack_bytes=pack_bytes, max_rows=max_rows)
    for begin in range(0, rows, limit):
        end = min(begin + limit, rows)
        n = end - begin
        keys = torch.zeros((n, width, 1, 512), dtype=data.dtype, device=data.device)
        keys[..., 0].copy_(data[begin:end].reshape(n, width, 1))
        query = torch.zeros((n, 1, 1, 512), dtype=data.dtype, device=data.device)
        query[..., 0] = 1
        qr = torch.zeros((n, 1, 1, 64), dtype=data.dtype, device=data.device)
        kr = torch.zeros((n, width, 1, 64), dtype=data.dtype, device=data.device)
        index = torch.arange(width, dtype=torch.int32, device=data.device).reshape(1, 1, 1, width).expand(n, 1, 1, width).contiguous()
        # All three original outputs are computed, even though only max is used.
        attention, maximum, total = ops.npu_sparse_flash_attention(query, keys, keys, index, 1.0,
            query_rope=qr, key_rope=kr, sparse_block_size=1, sparse_mode=0, attention_mode=2,
            layout_query='BSND', layout_kv='BSND', return_softmax_lse=True)
        out[begin:end].copy_(maximum.reshape(n).to(data.dtype))
        del attention, maximum, total
    return out


def _negate(data, muls_add):
    import torch
    shape = data.shape
    flat = data.contiguous().reshape(-1, 1)
    return muls_add(flat, torch.full_like(flat, -0.0), -1.0).reshape(shape)


def _selection_run(p, op, ops, muls_add=None, *, max_rows=MAX_STAT_ROWS, pack_bytes=PACK_BYTES):
    shape = tuple(p['output'][0])
    minimum = op in ('MinimumFwdOp', 'LogicalAndFwdOp')
    boolean = op in ('LogicalOrFwdOp', 'LogicalAndFwdOp')
    relu = op == 'ReluFwdOp'
    limit = _stat_row_limit(2, max_rows=max_rows, pack_bytes=pack_bytes)

    def run(x, other=None):
        import torch
        result = torch.empty(prod(shape), dtype=torch.bool if boolean else x.dtype, device=x.device)
        for begin in range(0, prod(shape), limit):
            end = min(begin + limit, prod(shape))
            a = _flat_slice(x, shape, begin, end)
            b = torch.zeros_like(a) if relu else _flat_slice(other, shape, begin, end)
            data = torch.stack((a, b), dim=1)
            if boolean:
                data = data.to(torch.bfloat16)  # exact 0/1 representation
            if minimum:
                data = _negate(data, muls_add)
            out = _stat_max(data, ops, max_rows=max_rows, pack_bytes=pack_bytes)
            if minimum:
                out = _negate(out, muls_add)
            result[begin:end].copy_(out.to(result.dtype))
        return result.reshape(shape)

    return run


def _reduction_run(p, op, ops, muls_add=None, *, max_rows=MAX_STAT_ROWS, pack_bytes=PACK_BYTES):
    order, width = reduction_geometry(p)
    minimum = op in ('AminFwdOp', 'AllFwdOp')
    boolean = op in ('AnyFwdOp', 'AllFwdOp')
    limit = _stat_row_limit(width, max_rows=max_rows, pack_bytes=pack_bytes)

    def run(x):
        import torch
        # Whole reduction domain remains in one sparse-attention call per row.
        data = x.permute(order).contiguous().reshape(-1, width)
        result = torch.empty(data.shape[0], dtype=torch.bool if boolean else x.dtype, device=x.device)
        for begin in range(0, data.shape[0], limit):
            end = min(begin + limit, data.shape[0])
            packed = data[begin:end].to(torch.bfloat16) if boolean else data[begin:end]
            if minimum:
                packed = _negate(packed, muls_add)
            out = _stat_max(packed, ops, max_rows=max_rows, pack_bytes=pack_bytes)
            if minimum:
                out = _negate(out, muls_add)
            result[begin:end].copy_(out.to(result.dtype))
        return result.reshape(p['output'][0])

    return run


def _pool_run(p, ops, *, adaptive=False, max_rows=MAX_STAT_ROWS, pack_bytes=PACK_BYTES):
    plan = pool_geometry(p, adaptive=adaptive)
    spatial_count = prod(plan['spatial'])
    out_count = prod(plan['out'])

    def run(x):
        import torch
        data = x.contiguous().reshape(-1, spatial_count)
        result = torch.empty((data.shape[0], out_count), dtype=x.dtype, device=x.device)
        # Group equal-length VALID windows, preserving every output coordinate.
        # No -inf padding is injected into a dot product or an all-empty domain.
        for start in range(0, out_count, 256):
            buckets = {}
            for position in range(start, min(start + 256, out_count)):
                window = pool_window(plan, position)
                buckets.setdefault(len(window), []).append((position, window))
            for width, windows in buckets.items():
                if width == 0:
                    for position, _ in windows:
                        result[:, position] = float('-inf')
                    continue
                limit = _stat_row_limit(width, max_rows=max_rows, pack_bytes=pack_bytes)
                for wstart in range(0, len(windows), limit):
                    group = windows[wstart:wstart + limit]
                    count = len(group)
                    out_index = torch.tensor([i for i, _ in group], dtype=torch.long, device=x.device)
                    in_index = torch.tensor([j for _, window in group for j in window], dtype=torch.long, device=x.device)
                    plane_limit = max(1, limit // count)
                    for begin in range(0, data.shape[0], plane_limit):
                        end = min(begin + plane_limit, data.shape[0])
                        packed = data[begin:end].index_select(1, in_index).reshape(-1, width)
                        out = _stat_max(packed, ops, max_rows=max_rows, pack_bytes=pack_bytes).reshape(end - begin, count)
                        result[begin:end].index_copy_(1, out_index, out)
        return result.reshape(p['output'][0])

    return run


def extrema(case):
    validate(case)
    from .native import native_ops
    op, p = case._call.signature.name, case.params
    ops, sources = native_ops('csrc/attention/sparse_flash_attention')
    # Include the shared original VF headers used by the statistics implementation.
    from .runtime import VENDOR
    sources += sorted((VENDOR / 'csrc/attention/common/op_kernel/arch35/vf').glob('*.h'))
    negate = None
    if op in ('MinimumFwdOp', 'AminFwdOp', 'AllFwdOp', 'LogicalAndFwdOp'):
        arithmetic, extra = module(TRITON_ROOT + 'muls_add.py', hardware=True)
        negate = arithmetic.muls_add_triton
        sources += extra
    if op in ('AminFwdOp', 'AnyFwdOp', 'AllFwdOp'):
        run = _reduction_run(p, op, ops, negate)
    elif 'Pool' in op:
        run = _pool_run(p, ops, adaptive=op.startswith('Adaptive'))
    else:
        run = _selection_run(p, op, ops, negate)
    return candidate(run, 'npu_sparse_flash_attention.softmax_max' + (' + muls_add_triton' if negate else ''),
        'Per-call complete row/window packing with Q basis coordinate0 and K data coordinate0; '
        'sparse attention supplies max. Minimum/All/And use original native negation; bool casts are lossless 0/1. '
        'Exclude spatial padding, preserve ceil/dilation/adaptive endpoints and every output. '
        'Bound only independent output rows; original kernels and unused attention outputs stay intact and timed.',
        [*_sources(), *sources], timing_mode='host_completed', limitations=(
            'CPU ideal-native simulation is not evidence of hardware MAX NaN propagation, Inf, subnormal or signed-zero behavior. '
            'The unchanged full oracle and native boundary tests remain mandatory; no finite-only filtering or tolerance changes.',))
