import torch
import torch.nn.functional as F
def selective_scan_ref(
    u,
    delta,
    A,
    B_in,
    C_in,
    D_in=None,
    z=None,
    delta_bias=None,
    seq_idx=None,
    initial_states=None,
    delta_softplus=True,
):
    """Mamba-1 S6 selective scan in float32, one step at a time: ``(y, final_states)``.

    The recurrence of ``mamba_ssm``'s ``selective_scan_ref``, with ``seq_idx`` restarting the
    state at every step whose id differs from the step before, and ``initial_states`` the
    state before the first step.
    """
    batch, dim, seqlen = u.shape
    groups = B_in.shape[1]
    uf = u.float()
    dt = delta.float() + (delta_bias.float()[:, None] if delta_bias is not None else 0)
    if delta_softplus:
        dt = F.softplus(dt)
    rep = dim // groups
    Bf = B_in.float().repeat_interleave(rep, 1)
    Cf = C_in.float().repeat_interleave(rep, 1)
    h = (
        initial_states.float().clone()
        if initial_states is not None
        else torch.zeros(batch, dim, A.shape[1], device=u.device)
    )
    exponents=dt[...,None]*A[None,:,None,:]
    decay=torch.pow(torch.full_like(exponents,2.718281828459045,dtype=torch.float64),exponents).float()
    ys = []
    for t in range(seqlen):
        if seq_idx is not None and t > 0:
            restart = seq_idx[:, t] != seq_idx[:, t - 1]
            h = torch.where(restart[:, None, None], 0.0, h)
        h = (
            decay[:, :, t] * h
            + (dt[:, :, t] * uf[:, :, t])[..., None] * Bf[..., t]
        )
        ys.append((h * Cf[..., t]).sum(-1))
    y = torch.stack(ys, -1)
    if D_in is not None:
        y = y + uf * D_in.float()[:, None]
    if z is not None:
        y = y * F.silu(z.float())
    return y.to(u.dtype), h
