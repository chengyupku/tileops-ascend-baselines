import torch
import torch.nn.functional as F
_REF_CHUNK=16


def gdn_ref(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    a_log=None,
    dt_bias=None,
    *,
    scale=None,
    raw_gate=False,
    beta_sigmoid=False,
    allow_neg_eigval=False,
    l2norm=False,
    state_v_first=False,
):
    """The delta rule evaluated independently of the chunked kernels and of FLA.

    A decode call, one token per sequence, runs the FP32 recurrence token by token. A
    longer call evaluates the same recurrence chunk by chunk in FP64: within a chunk the
    written values solve the unit lower-triangular system exactly, with no truncated
    inverse, and the outputs and final state follow in closed form. The sequence
    boundaries come from the call's offsets, never from the fixture's lengths, and every
    sequence still running advances in the same step.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        rows = list(range(q.shape[0]))
        starts, lengths = [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    else:
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        rows, starts = [0] * (len(bounds) - 1), bounds[:-1]
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    if raw_gate:
        g = -torch.exp(a_log) * torch.nn.functional.softplus(g.float() + dt_bias)
    if beta_sigmoid:
        beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
    longest = max(lengths, default=0)
    chunk, dtype = (1, torch.float32) if longest <= 1 else (64, torch.float32)
    # Longest first, so the sequences still running at a step are a prefix.
    order = sorted(range(len(lengths)), key=lambda sequence: -lengths[sequence])
    row = torch.tensor([rows[s] for s in order], dtype=torch.long, device=q.device)
    start = torch.tensor([starts[s] for s in order], dtype=torch.long, device=q.device)
    length = torch.tensor([lengths[s] for s in order], dtype=torch.long, device=q.device)
    if initial_state is None:
        state = torch.zeros(
            len(order), v.shape[2], q.shape[-1], v.shape[-1], dtype=dtype, device=q.device
        )
    else:
        state = initial_state[order].to(dtype)
        state = state.transpose(-1, -2).contiguous() if state_v_first else state
    output = torch.empty_like(v)
    offset = torch.arange(chunk, device=q.device)
    inclusive = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = inclusive.tril(-1)
    running = len(order)
    for first in range(0, longest, chunk):
        while lengths[order[running - 1]] <= first:
            running -= 1
        r = row[:running, None]
        valid = first + offset < length[:running, None]
        t = (start[:running, None] + first + offset).clamp(max=q.shape[1] - 1)
        mask = valid.to(dtype)[..., None]
        # [running, chunk, value heads, ...]; value head h reads the key head its group shares.
        q_c = q[r, t].to(dtype).repeat_interleave(group, dim=2)
        k_c = k[r, t].to(dtype).repeat_interleave(group, dim=2)
        if l2norm:
            q_c = q_c * torch.rsqrt(q_c.square().sum(-1, keepdim=True) + 1e-6)
            k_c = k_c * torch.rsqrt(k_c.square().sum(-1, keepdim=True) + 1e-6)
        live = state[:running]
        if chunk == 1:
            q_t, k_t = q_c[:, 0] * scale, k_c[:, 0]
            decay = g[r, t][:, 0].float().exp()
            old_value = torch.einsum("nhkv,nhk->nhv", live, k_t)
            value = beta[r, t][:, 0].float().unsqueeze(-1) * (
                v[r, t][:, 0].float() - decay.unsqueeze(-1) * old_value
            )
            live = decay[..., None, None] * live + k_t.unsqueeze(-1) * value.unsqueeze(-2)
            out = torch.einsum("nhk,nhkv->nhv", q_t, live)[:, None]
        else:
            # Padding carries no key, value or step size and no decay, so it leaves the state.
            q_c, k_c = (x.transpose(1, 2) for x in (q_c * scale, k_c * mask[..., None]))
            v_c = (v[r, t].to(dtype) * mask[..., None]).transpose(1, 2)
            log_decay = (g[r, t].to(dtype) * mask).transpose(1, 2).cumsum(-1)
            step = (beta[r, t].to(dtype) * mask).transpose(1, 2)
            gap = log_decay[..., :, None] - log_decay[..., None, :]
            system = (
                step[..., None]
                * gap.masked_fill(~strict, -torch.inf).exp()
                * (k_c @ k_c.transpose(-1, -2))
            )
            rhs = step[..., None] * (v_c - log_decay.exp()[..., None] * (k_c @ live))
            written = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
            attention = gap.masked_fill(~inclusive, -torch.inf).exp() * (
                q_c @ k_c.transpose(-1, -2)
            )
            out = (log_decay.exp()[..., None] * (q_c @ live) + attention @ written).transpose(1, 2)
            carry = (log_decay[..., -1:] - log_decay).exp()[..., None]
            live = (
                log_decay[..., -1].exp()[..., None, None] * live
                + (k_c * carry).transpose(-1, -2) @ written
            )
        state[:running] = live
        output[r.expand_as(t)[valid], t[valid]] = out[valid].to(q.dtype)
    final = torch.empty_like(state)
    final[order] = state
    final = final.transpose(-1, -2) if state_v_first else final
    return output, final.float()

def kda_ref(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    a_log=None,
    dt_bias=None,
    *,
    scale=None,
    raw_gate=False,
    lower_bound=None,
    beta_sigmoid=False,
    allow_neg_eigval=False,
    l2norm=False,
    state_v_first=False,
):
    """The KDA recurrence evaluated independently of the in-tree kernels and of FLA.

    Per token: the state decays by ``exp(g)`` along its key axis, the delta rule writes
    ``beta * (v - k^T S)`` along ``k``, and the output reads ``q^T S``. The recurrence runs
    chunk by chunk in FP64: within a chunk the written values solve the unit
    lower-triangular system exactly, with every per-channel decay gap materialized, so no
    gap is exponentiated with a positive argument. The sequence boundaries come from the
    call's offsets, and a sequence with no token ends on the state it started from.
    """
    dtype = torch.float32
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        rows = list(range(q.shape[0]))
        starts, lengths = [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    else:
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        rows, starts = [0] * (len(bounds) - 1), bounds[:-1]
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    g = g.to(dtype)
    if raw_gate:
        heads, dim = v.shape[2], q.shape[-1]
        x = g + (0.0 if dt_bias is None else dt_bias.to(dtype).view(heads, dim))
        rate = a_log.to(dtype).exp()[:, None]
        g = (
            lower_bound * torch.sigmoid(rate * x)
            if lower_bound is not None
            else -rate * F.softplus(x)
        )
    beta = beta.to(dtype)
    if beta_sigmoid:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    longest = max(lengths, default=0)
    # A decode call runs one token per step, and needs no gap tensor.
    chunk = 1 if longest <= 1 else _REF_CHUNK
    # Longest first, so the sequences still running at a step are a prefix.
    order = sorted(range(len(lengths)), key=lambda sequence: -lengths[sequence])
    row = torch.tensor([rows[s] for s in order], dtype=torch.long, device=q.device)
    start = torch.tensor([starts[s] for s in order], dtype=torch.long, device=q.device)
    length = torch.tensor([lengths[s] for s in order], dtype=torch.long, device=q.device)
    if initial_state is None:
        state = torch.zeros(
            len(order), v.shape[2], q.shape[-1], v.shape[-1], dtype=dtype, device=q.device
        )
    else:
        state = initial_state[order].to(dtype)
        state = state.transpose(-1, -2).contiguous() if state_v_first else state
    output = torch.zeros_like(v)
    offset = torch.arange(chunk, device=q.device)
    inclusive = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = inclusive.tril(-1)
    running = len(order)
    for first in range(0, longest, chunk):
        while lengths[order[running - 1]] <= first:
            running -= 1
        r = row[:running, None]
        valid = first + offset < length[:running, None]
        t = (start[:running, None] + first + offset).clamp(max=q.shape[1] - 1)
        mask = valid.to(dtype)[..., None]
        # [running, chunk, value heads, K]; value head h reads the key head its group shares.
        q_c = q[r, t].to(dtype).repeat_interleave(group, dim=2)
        k_c = k[r, t].to(dtype).repeat_interleave(group, dim=2)
        if l2norm:
            q_c = q_c * torch.rsqrt(q_c.square().sum(-1, keepdim=True) + 1e-6)
            k_c = k_c * torch.rsqrt(k_c.square().sum(-1, keepdim=True) + 1e-6)
        # [running, value heads, chunk, ...]. Padding carries no key, value, step or decay.
        q_c = (q_c * scale).transpose(1, 2)
        k_c = (k_c * mask[..., None]).transpose(1, 2)
        v_c = (v[r, t].to(dtype) * mask[..., None]).transpose(1, 2)
        step = (beta[r, t] * mask).transpose(1, 2)
        log_decay = (g[r, t] * mask[..., None]).transpose(1, 2).cumsum(2)
        live = state[:running]
        # gap[i, j, c] = log_decay[i, c] - log_decay[j, c], used only where j <= i.
        gap = log_decay[..., :, None, :] - log_decay[..., None, :, :]
        weight = gap.masked_fill(~inclusive[..., None], -torch.inf).exp()
        system = step[..., None] * torch.einsum("nhic,nhijc,nhjc->nhij", k_c, weight, k_c)
        system = system.masked_fill(~strict, 0.0)
        decayed = log_decay.exp()
        rhs = step[..., None] * (v_c - (k_c * decayed) @ live)
        written = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        attention = torch.einsum("nhic,nhijc,nhjc->nhij", q_c, weight, k_c)
        out = ((q_c * decayed) @ live + attention @ written).transpose(1, 2)
        carry = (log_decay[..., -1:, :] - log_decay).exp()
        live = decayed[..., -1, :, None] * live + (k_c * carry).transpose(-1, -2) @ written
        state[:running] = live
        output[r.expand_as(t)[valid], t[valid]] = out[valid].to(v.dtype)
    final = torch.empty_like(state)
    final[order] = state
    final = final.transpose(-1, -2) if state_v_first else final
    return output, final.float()