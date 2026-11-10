"""Device counterparts of frozen CPU references. Original formulas/checkers retained; FP32 replaces unsupported FP64 on device. Generated methods are frozen separately, not modifications of CPU oracle."""

import torch, math

import torch.nn.functional as F

from workloads.ascend_extended_transform_quant import _HADAMARD_SIGNS

from workloads.ascend_extended_dsa import _LATENT,_ROPE,_GROUP,_GROUPS,_SCALE_END,_ROW_BYTES

def CausalConvCall_device(self, x, weight, conv_state, cu_seqlens, bias=None, has_initial_state=None):
    device, dtype = (x.device, x.dtype)
    xx, ww, state = (x.float(), weight.float(), conv_state.clone())
    offsets = cu_seqlens.tolist()
    bb = None if bias is None else bias.float()
    flags = None if has_initial_state is None else has_initial_state.tolist()
    width = weight.shape[-1]
    result = torch.empty_like(xx)
    for i, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        history = state[i].float() if flags is not None and flags[i] else torch.zeros_like(state[i]).float()
        tokens = torch.cat((history, xx[:, begin:end]), dim=-1)
        y = F.conv1d(tokens.unsqueeze(0), ww.unsqueeze(1), bias=bb, groups=x.shape[0])[0]
        if self.call.params['activation'] == 'silu':
            y = F.silu(y)
        result[:, begin:end] = y
        state[i].copy_(tokens[:, -(width - 1):])
    return (result.to(device=device, dtype=dtype), state.to(device))

def CausalConvUpdateCall_device(self, x, weight, conv_state, bias=None):
    state = torch.cat((conv_state[..., 1:], x.unsqueeze(-1)), dim=-1)
    result = (state.float() * weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:
        result += bias.float()
    if self.call.params['activation'] == 'silu':
        result = F.silu(result)
    return (result.to(device=x.device, dtype=x.dtype), state.to(x.device))

def PagedCacheGatherCall_device(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
    source = cache
    if source.dtype == torch.float8_e4m3fn:
        source = source.float()
    table = block_table.long()
    offsets = cu_seq_lens.tolist()
    starts = [0] * (len(offsets) - 1) if seq_starts is None else seq_starts.tolist()
    output = torch.empty(dst.shape, dtype=dst.dtype, device=dst.device)
    page_size = source.shape[1]
    for batch, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        positions = torch.arange(starts[batch], starts[batch] + end - begin, device=dst.device)
        page_ids = table[batch, positions // page_size]
        values = source[page_ids, positions % page_size]
        if scale is not None:
            values = values.float() * scale.float()
        output[begin:end] = values.to(dst.dtype)
    return output.to(dst.device)

def Int8GemmCall_device(self, a, b, scale_a, scale_b, bias=None):
    result = torch._int_mm(a, b.T.contiguous()).float()
    result *= scale_a.float()
    result *= scale_b.float()
    if bias is not None:
        result += bias.float()
    return result.to(device=a.device, dtype=getattr(torch, self.call.params['out_dtype']))

def MergeAttentionCall_device(self, v_a, s_a, v_b, s_b):
    a, b = (s_a.float(), s_b.float())
    maximum = torch.maximum(a, b)
    empty = torch.isneginf(maximum)
    safe_maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
    wa, wb = (torch.exp(a - safe_maximum), torch.exp(b - safe_maximum))
    denominator = wa + wb
    va = torch.where((wa != 0).unsqueeze(-1), v_a.float(), 0.0)
    vb = torch.where((wb != 0).unsqueeze(-1), v_b.float(), 0.0)
    result = (va * wa.unsqueeze(-1) + vb * wb.unsqueeze(-1)) / torch.where(empty, torch.ones_like(denominator), denominator).unsqueeze(-1)
    lse = safe_maximum + torch.log(denominator)
    return (result.to(device=v_a.device, dtype=v_a.dtype), lse.to(device=s_a.device, dtype=s_a.dtype))

def SelectiveScanCall_device(self, u, delta, A, B_in, C_in, D_in=None, z=None, delta_bias=None, seq_idx=None, initial_states=None):
    batch, channels, length = u.shape
    groups, states = B_in.shape[1:3]
    heads = channels // groups
    source = u.float().reshape(batch, groups, heads, length)
    step = delta.float().reshape_as(source)
    if delta_bias is not None:
        step = step + delta_bias.float().reshape(1, groups, heads, 1)
    if self.call.params['delta_softplus']:
        step = F.softplus(step)
    decay = A.float().reshape(1, groups, heads, states)
    b, c = (B_in.float(), C_in.float())
    state = torch.zeros(batch, groups, heads, states, dtype=torch.float32, device=u.device) if initial_states is None else initial_states.float().reshape(batch, groups, heads, states).clone()
    sequence = None if seq_idx is None else seq_idx
    output = torch.empty(batch, groups, heads, length, dtype=torch.float32, device=u.device)
    for token in range(length):
        if sequence is not None and token:
            reset = sequence[:, token] != sequence[:, token - 1]
            state[reset] = 0
        dt = step[..., token, None]
        state = torch.exp(dt * decay) * state + dt * b[:, :, None, :, token] * source[..., token, None]
        output[..., token] = (state * c[:, :, None, :, token]).sum(-1)
    output = output.reshape(batch, channels, length)
    if D_in is not None:
        output += u.float() * D_in.float().reshape(1, channels, 1)
    if z is not None:
        output *= F.silu(z.float())
    return (output.to(device=u.device, dtype=u.dtype), state.reshape(batch, channels, states).to(device=u.device, dtype=torch.float32))

def FusedQKNormRoPECall_device(self, qkv, q_weight, k_weight, cos_sin_cache, positions):
    p = self.call.params
    width = q_weight.numel()
    heads, kv_heads = (p['num_heads'], p['num_kv_heads'])
    output = qkv.clone().reshape(qkv.shape[0], heads + 2 * kv_heads, width)
    cache = cos_sin_cache.float()[positions.long()]
    rotary = cache.shape[-1]
    cos, sin = (part[:, None, :] for part in cache.chunk(2, dim=-1))
    for start, count, weight in ((0, heads, q_weight), (heads, kv_heads, k_weight)):
        value = output[:, start:start + count].float()
        value = value * (torch.rsqrt(value.square().mean(-1, keepdim=True) + p['eps']) * weight.float())
        rotated = value[..., :rotary]
        if p['rope_layout'] == 'neox':
            first, second = rotated.chunk(2, dim=-1)
            rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        else:
            first, second = (rotated[..., ::2], rotated[..., 1::2])
            rotated = torch.stack((first * cos - second * sin, second * cos + first * sin), dim=-1).flatten(-2)
        value[..., :rotary] = rotated
        output[:, start:start + count] = value.to(qkv.dtype)
    return output.reshape_as(qkv).to(qkv.device)

def FusedAddRMSNormQuantCall_device(self, x, residual, weight):
    p = self.call.params
    value = x.float() + residual.float()
    residual_out = value.to(x.dtype)
    inv_rms = torch.rsqrt(value.square().mean(-1, keepdim=True) + p['eps'])
    normalized = ((value * inv_rms).to(x.dtype) * weight).float()
    dtype = getattr(torch, p['quant_dtype'])
    maximum = 127.0 if dtype == torch.int8 else torch.finfo(dtype).max
    scale = normalized.abs().amax(-1, keepdim=True) / maximum
    if p['scale_ub'] is not None:
        scale = scale.clamp(max=p['scale_ub'])
    divided = normalized / torch.where(scale == 0, torch.ones_like(scale), scale)
    if dtype == torch.int8:
        divided = divided.round()
    minimum = -128.0 if dtype == torch.int8 else -maximum
    quantized = divided.clamp(minimum, maximum).to(dtype)
    return (quantized.to(x.device), scale.to(x.device), residual_out.to(x.device))

def MLAPagedCall_device(self, q, kv_cache, block_table, cache_seqlens, kv_scale=None):
    p = self.call.params
    batch, query_length, heads, width = q.shape
    rank = p['kv_lora_rank']
    scale = p['sm_scale'] if p['sm_scale'] is not None else width ** (-0.5)
    query = q.detach().float()
    cache = kv_cache.detach()
    table = block_table.detach().long()
    lengths = cache_seqlens.detach().tolist()
    dequant = None if kv_scale is None else kv_scale.detach().float()
    output = torch.empty((batch, query_length, heads, rank), dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, query_length, heads), dtype=torch.float32, device=q.device)
    page_size = cache.shape[1]
    for b, length in enumerate(lengths):
        pages = table[b, :(length + page_size - 1) // page_size]
        if cache.dtype == torch.float8_e4m3fn:
            rows = cache.view(torch.uint8).index_select(0, pages).view(cache.dtype)
        else:
            rows = cache.index_select(0, pages)
        key = rows.reshape(-1, width)[:length].float()
        if dequant is not None:
            key = key * dequant
        value = key[:, :rank]
        for start in range(0, query_length, 16):
            end = min(start + 16, query_length)
            scores = torch.matmul(query[b, start:end], key.T) * scale
            if p['is_causal']:
                q_positions = length - query_length + torch.arange(start, end, device=q.device)
                k_positions = torch.arange(length, device=q.device)
                visible = k_positions[None, :] <= q_positions[:, None]
                scores.masked_fill_(~visible[:, None, :], -math.inf)
            output[b, start:end] = torch.matmul(scores.softmax(-1), value).to(q.dtype)
            lse[b, start:end] = scores.logsumexp(-1) / math.log(2.0)
    return (output.to(q.device), lse.to(q.device))

def MLAVarlenCall_device(self, q, k_nope, k_pe, v, cu_seqlens):
    p = self.call.params
    scale = p['sm_scale'] if p['sm_scale'] is not None else q.shape[-1] ** (-0.5)
    query, nope, pe, value = (t.detach() for t in (q, k_nope, k_pe, v))
    offsets = cu_seqlens.detach().tolist()
    total, heads = q.shape[:2]
    output = torch.empty((total, heads, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((total, heads), dtype=torch.float32, device=q.device)
    for begin, end in zip(offsets[:-1], offsets[1:], strict=True):
        length = end - begin
        pe_part = pe[begin:end].float()
        for h0 in range(0, heads, 4):
            h1 = min(h0 + 4, heads)
            key = torch.cat((nope[begin:end, h0:h1].float(), pe_part[:, None, :].expand(-1, h1 - h0, -1)), dim=-1).permute(1, 2, 0)
            values = value[begin:end, h0:h1].float().transpose(0, 1)
            for r0 in range(0, length, 128):
                r1 = min(r0 + 128, length)
                key_end = r1 if p['is_causal'] else length
                queries = query[begin + r0:begin + r1, h0:h1].float().transpose(0, 1)
                scores = torch.matmul(queries, key[..., :key_end]) * scale
                if p['is_causal']:
                    visible = torch.arange(key_end, device=q.device)[None, :] <= torch.arange(r0, r1, device=q.device)[:, None]
                    scores.masked_fill_(~visible[None, :, :], -math.inf)
                result = torch.matmul(scores.softmax(-1), values[:, :key_end])
                output[begin + r0:begin + r1, h0:h1] = result.transpose(0, 1).to(q.dtype)
                lse[begin + r0:begin + r1, h0:h1] = scores.logsumexp(-1).T
    return (output.to(q.device), lse.to(q.device))

def MLAKVCacheWriteCall_device(self, kv_c, k_pe, kv_cache, slot_mapping, scale=None, positions=None, cos_sin_cache=None):
    output = kv_cache.detach().clone()
    flat = output.view(-1, output.shape[-1])
    slots = slot_mapping.detach().long()
    if ((slots < -1) | (slots >= flat.shape[0])).any().item():
        raise ValueError('MLA cache slots must be -1 or valid physical row indices')
    tokens = (slots >= 0).nonzero().flatten()
    destinations = slots[tokens]
    if destinations.unique().numel() != destinations.numel():
        raise ValueError('MLA cache writes require distinct non-negative slots')
    latent = kv_c.detach().index_select(0, tokens)
    positional = k_pe.detach().index_select(0, tokens)
    if self.call.params['fuse_rope']:
        selected_positions = positions.detach().long().index_select(0, tokens)
        trig = cos_sin_cache.detach().index_select(0, selected_positions).float()
        cosine, sine = trig.chunk(2, dim=-1)
        positional = positional.float()
        layout = self.call.params['rope_layout']
        if layout == 'neox':
            first, second = positional.chunk(2, dim=-1)
            positional = torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)
        elif layout == 'interleaved':
            first, second = (positional[:, ::2], positional[:, 1::2])
            positional = torch.stack((first * cosine - second * sine, second * cosine + first * sine), dim=-1).flatten(-2)
        else:
            raise ValueError(f'Unknown MLA RoPE layout: {layout!r}')
        positional = positional.to(k_pe.dtype)
    stored = torch.cat((latent, positional), dim=-1)
    if output.dtype == torch.float8_e4m3fn:
        limit = torch.finfo(output.dtype).max
        stored = (stored.float() / scale.detach().float()).clamp(-limit, limit)
    stored = stored.to(output.dtype)
    flat.view(torch.uint8).index_copy_(0, destinations, stored.contiguous().view(torch.uint8))
    return output.to(kv_cache.device)

def _decode_rows(rows):
    """Decode selected CPU uint8 rows; preserve the V3.2 BF16 storage rounding."""
    codes = rows[:, :_LATENT].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[:, _LATENT:_SCALE_END].contiguous().view(torch.float32)
    latent = codes.reshape(-1, _GROUPS, _GROUP) * scales.unsqueeze(-1)
    latent = latent.reshape(-1, _LATENT).to(torch.bfloat16)
    rope = rows[:, _SCALE_END:].contiguous().view(torch.bfloat16)
    return torch.cat((latent, rope), dim=-1).float()

def DSAPagedCall_device(self, q, kv_cache, block_table, cache_seqlens, indices):
    """Evaluate only the supplied selected rows, without caching input-dependent data.

        All decoding, indexing and attention arithmetic run on CPU. Duplicated
        indices contribute once per occurrence. Only -1 is padding; other
        out-of-range indices violate the manifest rather than silently masking.
        The selection is the entire mask: do not impose a second causal mask.
        """
    if q.ndim != 4 or q.shape[-1] != _LATENT + _ROPE or q.dtype != torch.bfloat16:
        raise ValueError('q must be bfloat16 [B, S_q, H, 576]')
    if kv_cache.ndim != 3 or kv_cache.shape[-1] != _ROW_BYTES or kv_cache.dtype != torch.uint8 or (not kv_cache.is_contiguous()):
        raise ValueError('kv_cache must be contiguous uint8 [NP, PS, 656]')
    batch, queries, heads, _ = q.shape
    pages, page_size, _ = kv_cache.shape
    if block_table.ndim != 2 or block_table.shape[0] != batch or block_table.dtype != torch.int32:
        raise ValueError('block_table must be int32 [B, W]')
    if cache_seqlens.shape != (batch,) or cache_seqlens.dtype != torch.int32:
        raise ValueError('cache_seqlens must be int32 [B]')
    if indices.ndim != 3 or indices.shape[:2] != (batch, queries) or indices.dtype != torch.int32:
        raise ValueError('indices must be int32 [B, S_q, K]')
    if queries <= 0 or page_size <= 0:
        raise ValueError('S_q and PS must be positive')
    scale = self.call.params['sm_scale']
    scale = (_LATENT + _ROPE) ** (-0.5) if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError('sm_scale must be finite')
    query = q.detach().float()
    cache = kv_cache.detach()
    table = block_table.detach().long()
    lengths = cache_seqlens.detach().long()
    selected = indices.detach().long()
    if bool(((table < 0) | (table >= pages)).any()):
        raise ValueError('block_table entries must lie in [0, NP)')
    if bool(((lengths < queries) | (lengths > table.shape[1] * page_size)).any()):
        raise ValueError('cache_seqlens entries must lie in [S_q, W * PS]')
    invalid = (selected != -1) & ((selected < 0) | (selected >= lengths[:, None, None]))
    if bool(invalid.any()):
        raise ValueError('each index must be -1 or a position below cache_seqlens[b]')
    output = torch.zeros((batch, queries, heads, _LATENT), dtype=torch.bfloat16, device=q.device)
    lse = torch.full((batch, queries, heads), float('inf'), dtype=torch.float32, device=q.device)
    for b in range(batch):
        for s in range(queries):
            positions = selected[b, s]
            positions = positions[positions != -1]
            if positions.numel() == 0:
                continue
            physical_pages = table[b, positions // page_size]
            rows = cache[physical_pages, positions % page_size]
            key = _decode_rows(rows)
            scores = query[b, s] @ key.T * scale
            row_lse = torch.logsumexp(scores, dim=-1)
            probabilities = torch.exp(scores - row_lse.unsqueeze(-1))
            output[b, s] = (probabilities @ key[:, :_LATENT]).to(torch.bfloat16)
            lse[b, s] = row_lse.to(torch.float32)
    return (output.to(q.device), lse.to(q.device))

def HadamardTransformCall_device(self, x):
    base_order = self.call.params['base_order']
    if base_order not in (1, 12, 20, 28, 40):
        raise ValueError(f'Unsupported Hadamard base_order: {base_order}')
    if x.ndim < 1 or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('Hadamard requires a non-scalar float16/bfloat16 tensor')
    n = x.shape[-1]
    if n <= 0 or n % base_order:
        raise ValueError('Hadamard n must be a positive multiple of base_order')
    q = n // base_order
    if q & q - 1 or q > 268435456:
        raise ValueError('Hadamard n/base_order must be a power of two <= 2**28')
    source = x.detach().to(device=x.device).contiguous().reshape(-1, n)
    output = torch.empty(source.shape, dtype=x.dtype, device=x.device)
    base_matrix = None
    if base_order != 1:
        base_matrix = torch.tensor([[1 if sign == '+' else -1 for sign in row] for row in _HADAMARD_SIGNS[base_order]], dtype=torch.float32, device=x.device)
    rows_per_chunk = max(1, (1 << 20) // n)
    normalization = 1.0 / math.sqrt(n)
    for begin in range(0, source.shape[0], rows_per_chunk):
        end = min(begin + rows_per_chunk, source.shape[0])
        rows = end - begin
        values = source[begin:end].to(torch.float32).reshape(rows, base_order, q)
        width = 1
        while width < q:
            pairs = values.reshape(rows, base_order, q // (2 * width), 2, width)
            left, right = (pairs[..., 0, :].clone(), pairs[..., 1, :].clone())
            pairs[..., 0, :].copy_(left + right)
            pairs[..., 1, :].copy_(left - right)
            width *= 2
        if base_matrix is not None:
            values = torch.matmul(base_matrix, values)
        output[begin:end].copy_((values.reshape(rows, n) * normalization).to(dtype=x.dtype))
    return output.reshape(x.shape).to(device=x.device)

def install():

 import workloads.ascend_extended as m

 m.CausalConvCall.ref_program=CausalConvCall_device

 m.CausalConvUpdateCall.ref_program=CausalConvUpdateCall_device

 m.PagedCacheGatherCall.ref_program=PagedCacheGatherCall_device

 m.Int8GemmCall.ref_program=Int8GemmCall_device

 m.MergeAttentionCall.ref_program=MergeAttentionCall_device

 m.SelectiveScanCall.ref_program=SelectiveScanCall_device

 m.FusedQKNormRoPECall.ref_program=FusedQKNormRoPECall_device

 m.FusedAddRMSNormQuantCall.ref_program=FusedAddRMSNormQuantCall_device

 import workloads.ascend_extended_attention as m

 m.MLAPagedCall.ref_program=MLAPagedCall_device

 m.MLAVarlenCall.ref_program=MLAVarlenCall_device

 m.MLAKVCacheWriteCall.ref_program=MLAKVCacheWriteCall_device

 import workloads.ascend_extended_dsa as m

 m.DSAPagedCall.ref_program=DSAPagedCall_device

 import workloads.ascend_extended_transform_quant as m

 m.HadamardTransformCall.ref_program=HadamardTransformCall_device
