"""User-authorized six-output torch reference, independent of the CPU oracle.

All arithmetic stays on the inputs' device. The inverse is the finite Neumann
series of a strictly lower triangular matrix, evaluated by repeated doubling.
This is a new reference, not a claim about the historical single-output baseline.
"""
import torch


@torch.no_grad()
def reference(q, k, v, beta, chunk_size):
    dtype, device = q.dtype, q.device
    B, H, L, K = q.shape
    V, C = v.shape[-1], chunk_size
    assert L > 0 and C > 0 and L % C == 0
    assert all(x.device == device for x in (k, v, beta))
    N = L // C
    q = q.float().reshape(B, H, N, C, K)
    k = k.float().reshape(B, H, N, C, K)
    v = v.float().reshape(B, H, N, C, V)
    beta = beta.float().reshape(B, H, N, C, 1)
    power = -torch.tril(beta * (k @ k.transpose(-1, -2)), diagonal=-1)
    eye = torch.eye(C, dtype=torch.float32, device=device)
    inverse = eye + power
    degree = 1
    while degree < C - 1:
        power = power @ power
        inverse = inverse + power @ inverse
        degree = 2 * degree + 1
    w = inverse @ (beta * k)
    u = inverse @ (beta * v)
    state = torch.zeros((B, H, K, V), dtype=torch.float32, device=device)
    states, outputs = [state], []
    for i in range(N):
        qc, kc = q[:, :, i], k[:, :, i]
        residual = u[:, :, i] - w[:, :, i] @ state
        outputs.append(qc @ state + torch.tril(qc @ kc.transpose(-1, -2)) @ residual)
        state = state + kc.transpose(-1, -2) @ residual
        states.append(state)
    a = inverse.reshape(B, H, L, C).to(dtype).contiguous()
    return (torch.cat(outputs, dim=2).to(dtype).contiguous(),
            torch.stack(states, dim=2).contiguous(), a, a.clone(),
            w.reshape(B, H, L, K).to(dtype).contiguous(),
            u.reshape(B, H, L, V).to(dtype).contiguous())
