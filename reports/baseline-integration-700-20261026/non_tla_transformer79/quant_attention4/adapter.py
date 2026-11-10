"""Private, unvalidated top-level mode-6 adapter. Import is stdlib-only.
Run function only inside supervisor's NPU guard. No arithmetic on Q/K/V/scales.
All allocation, copies, metadata, attention and selection belong inside timing.
"""
CASES = {
 'llama-70b-fp8-q-lt-kv-float8_e4m3fn-bfloat16': ([512,512], [1024,2048],64,8,128),
 'llama-8b-fp8-float8_e4m3fn-bfloat16': ([512]*4,[512]*4,32,8,128),
 'qwen3-fp8-ragged-float8_e4m3fn-bfloat16': ([1,17,1000,0,1999],[1,17,1000,3000,2128],32,4,128),
}

def plan(case_id):
    ql,kl,h,hk,d = CASES[case_id]
    b=len(ql); pages=[(n+127)//128 for n in kl]
    return dict(q_lens=ql,kv_lens=kl,h=h,hk=hk,d=d,b=b,groups=h//hk,
                expanded_h=b*h,expanded_hk=b*hk,pages=pages,
                q_shape=[b*h,sum(ql),d],kv_shape=[sum(pages),b*hk,128,d],
                qscale_shape=[b*h,sum(ql)],kscale_shape=[sum(pages),b*hk,128],
                vscale_shape=[b*hk],out_shape=[sum(ql),b*h,d])

def run(case_id,q,k,v,cu_seqlens_q,cu_seqlens_kv,q_scale,k_scale,v_scale,*,sm_scale=None):
    # Lazy imports intentionally NOT exercised by the CPU worker.
    import torch
    import cann_ops_transformer as cot
    p=plan(case_id); b,h,hk,d=p['b'],p['h'],p['hk'],p['d']
    assert p['expanded_h']<=256 and p['expanded_hk']<=256
    assert tuple(q.shape)==(sum(p['q_lens']),h,d)
    assert tuple(k.shape)==tuple(v.shape)==(sum(p['kv_lens']),hk,d)
    assert all(t.dtype==torch.float8_e4m3fn for t in (q,k,v))
    assert all(t.dtype==torch.float32 and tuple(t.shape)==(b,hk) for t in (q_scale,k_scale,v_scale))
    # Read runtime lengths, never assume matching shapes imply matching segments.
    # These CPU transfers and synchronization must be included in adapter timing.
    actual_q=cu_seqlens_q.detach().cpu().tolist()
    actual_k=cu_seqlens_kv.detach().cpu().tolist()
    assert [y-x for x,y in zip(actual_q,actual_q[1:])]==p['q_lens']
    assert [y-x for x,y in zip(actual_k,actual_k[1:])]==p['kv_lens']
    qq=q.new_zeros(p['q_shape']); kk=k.new_zeros(p['kv_shape']); vv=v.new_zeros(p['kv_shape'])
    qs=q_scale.new_ones(p['qscale_shape']); ks=k_scale.new_ones(p['kscale_shape'])
    vs=v_scale.contiguous().reshape(-1)
    bt=torch.zeros((b,max(p['pages'])),dtype=torch.int32,device=q.device)
    page0=0
    for batch,(nq,nk,npages) in enumerate(zip(p['q_lens'],p['kv_lens'],p['pages'])):
        q0,k0=actual_q[batch],actual_k[batch]
        hs=slice(batch*h,(batch+1)*h); khs=slice(batch*hk,(batch+1)*hk)
        qq[hs,q0:q0+nq]=q[q0:q0+nq].permute(1,0,2)
        qs[hs,q0:q0+nq]=q_scale[batch].repeat_interleave(p['groups'])[:,None]
        for j in range(npages):
            n=min(128,nk-128*j); src=slice(k0+128*j,k0+128*j+n)
            kk[page0+j,khs,:n]=k[src].permute(1,0,2)
            vv[page0+j,khs,:n]=v[src].permute(1,0,2)
            ks[page0+j,khs,:n]=k_scale[batch,:,None]
            bt[batch,j]=page0+j
        page0+=npages
    sq=torch.tensor(p['q_lens'],dtype=torch.int32,device=q.device)
    sk=torch.tensor(p['kv_lens'],dtype=torch.int32,device=q.device)
    common=dict(quant_mode=6,cu_seqlens_q=cu_seqlens_q,seqused_q=sq,seqused_kv=sk,
                mask_mode=3,max_seqlen_q=max(p['q_lens']),max_seqlen_kv=max(p['kv_lens']),
                layout_q='NTD',layout_q_descale='NT',layout_kv='PA_BNBD',layout_out='TND')
    metadata=cot.quant_flash_attn_metadata(num_heads_q=b*h,num_heads_kv=b*hk,head_dim=d,**common)
    mask=torch.triu(torch.ones((2048,2048),dtype=torch.int8,device=q.device),diagonal=1)
    # Exactly one full attention call, with original optional metadata entrypoint.
    out,_=cot.quant_flash_attn(qq,kk,vv,qs,ks,vs,block_table=bt,
        p_scale=q_scale.new_ones((1,)),metadata=metadata,attn_mask=mask,
        softmax_scale=d**-0.5 if sm_scale is None else sm_scale,
        return_softmax_lse=False,**common)
    assert out.dtype==torch.bfloat16
    result=out.new_empty((sum(p['q_lens']),h,d))
    for batch,nq in enumerate(p['q_lens']):
        q0=actual_q[batch]
        result[q0:q0+nq]=out[q0:q0+nq,batch*h:(batch+1)*h]
    return result
