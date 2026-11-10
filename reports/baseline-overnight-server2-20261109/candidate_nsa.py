import torch
from einops import einsum, repeat
from adapters.execution_boundary import ExecutionBoundary
def ref_program(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    offsets: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    _ = token_indices
    dtype, device = q.dtype, q.device
    bs = self.block_size
    scale = self.scale if self.scale is not None else k.shape[-1] ** -0.5
    group = q.shape[1] // k.shape[1]
    n_slot = block_indices.shape[-1] * bs

    k, v, block_indices = (
        repeat(x, "t h d -> t (h g) d", g=group) for x in (k, v, block_indices)
    )
    block_counts = repeat(block_counts, "t h -> t (h g)", g=group)
    q, k, v = (x.float() for x in (q, k, v))
    heads = q.shape[1]
    # Which selected block each gathered position came from, for the count mask.
    slot_block = (torch.arange(n_slot, device=device) // bs).view(1, n_slot, 1)
    head = torch.arange(heads, device=device)

    o_slc = torch.zeros_like(v)
    for i in range(len(offsets) - 1):
        bos, eos = offsets[i].item(), offsets[i + 1].item()
        n_token = eos - bos
        q_b, k_b, v_b = q[bos:eos], k[bos:eos], v[bos:eos]

        # [t, s*bs, hq]: the token each selected block contributes, per head.
        pos = block_indices[bos:eos].unsqueeze(-1) * bs + torch.arange(bs, device=device)
        pos = pos.view(n_token, heads, n_slot).transpose(1, 2)
        # Out-of-range positions are masked below; clamping only keeps the gather legal.
        flat = (pos.clamp(0, n_token - 1) * heads + head).flatten()
        k_slc, v_slc = (x.reshape(-1, x.shape[-1]).index_select(0, flat).reshape(*pos.shape, x.shape[-1]) for x in (k_b, v_b))

        # A causal token sees keys up to itself, a non-causal one up to the sequence end.
        i_q = torch.arange(n_token, device=device).view(n_token, 1, 1)
        beyond = pos > i_q if self.is_causal else pos >= n_token
        attn = (
            einsum(q_b * scale, k_slc, "t h d, t n h d -> t n h")
            .masked_fill(
                (pos < 0) | beyond | (slot_block >= block_counts[bos:eos].unsqueeze(1)),
                float("-inf"),
            )
            .softmax(1)
            .nan_to_num(0.0)  # a token no selected block gives a key outputs zeros
        )
        o_slc[bos:eos] = einsum(attn, v_slc, "t n h, t n h v -> t h v") * self.g_slc[
            0, bos:eos
        ].unsqueeze(-1)

    return o_slc.to(dtype)

def bind(case,args,previous):
 assert case._call.signature.name=='NSAVarlenFwdOp'
 def run(*a):return ref_program(case.workload,*a)
 return ExecutionBoundary(run,tuple(args),lambda:None,{'candidate':'nsa-flat-index-v1','change':'same clamped token/head coordinates flattened to index_select; all original attention and masking preserved','input_dependent_work_timed':True})
