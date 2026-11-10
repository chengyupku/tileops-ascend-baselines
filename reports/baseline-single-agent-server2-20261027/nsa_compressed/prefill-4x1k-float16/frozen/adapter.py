"""Private, unvalidated single-original-FlashAttn adapter. Import is CPU safe.
All packing, metadata, copies and output conversion belong inside measured run.
No oracle/checker changes. Runtime dependencies are injected by deployment owner.
"""
def compressed_plan(offsets, chunk_offsets, bs):
    groups = []
    for b, (start, stop) in enumerate(zip(offsets, offsets[1:])):
        length = stop - start
        for closed in range(1, length // bs + 1):
            lo = start + closed * bs - 1
            hi = min(stop, start + (closed + 1) * bs - 1)
            groups.append((lo, hi, chunk_offsets[b], chunk_offsets[b] + closed))
    return groups


def compressed(torch, flash_attn, flash_attn_metadata, q, k, v,
               offsets, chunk_offsets, token_indices, *, bs, scale):
    groups = compressed_plan(offsets.tolist(), chunk_offsets.tolist(), bs)
    out = q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    lse = q.new_zeros((q.shape[0], q.shape[1]))
    if not groups:
        return out, lse
    qp = torch.cat([q[a:b] for a,b,c,d in groups]).contiguous()
    kp = torch.cat([k[c:d] for a,b,c,d in groups]).contiguous()
    vp = torch.cat([v[c:d] for a,b,c,d in groups]).contiguous()
    cq, ck = [0], [0]
    for a,b,c,d in groups:
        cq.append(cq[-1]+b-a)
        ck.append(ck[-1]+d-c)
    cuq = torch.tensor(cq, dtype=torch.int32, device=q.device)
    cuk = torch.tensor(ck, dtype=torch.int32, device=q.device)
    options = dict(cu_seqlens_q=cuq, cu_seqlens_kv=cuk,
                   max_seqlen_q=max(b-a for a,b,c,d in groups),
                   max_seqlen_kv=max(d-c for a,b,c,d in groups),
                   mask_mode=0, layout_q='TND', layout_kv='TND', layout_out='TND')
    meta = flash_attn_metadata(q.shape[1], k.shape[1], q.shape[2],
                              head_dim_v=v.shape[2], **options)
    op, lp = flash_attn(qp,kp,vp,metadata=meta,softmax_scale=scale,
                       return_softmax_lse=True,**options)
    # Original TND LSE is [heads, total_queries]; contract is [tokens, heads].
    lp = lp.transpose(0,1).to(q.dtype)
    for i,(a,b,c,d) in enumerate(groups):
        out[a:b] = op[cq[i]:cq[i+1]]
        lse[a:b] = lp[cq[i]:cq[i+1]]
    return out,lse
