"""Unregistered original FlashAttn adapters, only the four authorized exactkeys."""
SPECS={
 ('GQAPagedFwdOp','gpt-oss-sink-mixed-bfloat16-bfloat16'):dict(h=64,hkv=8,d=64,b=4,mq=256,mkv=1025,left=128),
 ('GQAPagedFwdOp','llama-8b-sliding-window-float16-float16'):dict(h=32,hkv=8,d=128,b=32,mq=1,mkv=16384,left=4096),
 ('GQAVarlenFwdOp','gpt-oss-sink-mixed-bfloat16'):dict(h=64,hkv=8,d=64,b=4,mq=1024,mkv=1025,left=128),
 ('GQAVarlenFwdOp','gpt-oss-sink-mixed-float16'):dict(h=64,hkv=8,d=64,b=4,mq=1024,mkv=1025,left=128),
}
def build_run(case):
    import torch
    from cann_ops_transformer.ops import flash_attn,flash_attn_metadata
    op=case._call.signature.name;s=SPECS[(op,case.id)]
    assert case.params['is_causal'] and case.params['window_size_left']==s['left']
    assert case.params['window_size_right']==-1
    # Same constant compressed mask as original tests/assets/impl/inputs.py.
    mask=torch.triu(torch.ones(2048,2048),diagonal=1).to(dtype=torch.int8,device='npu')
    base=dict(batch_size=s['b'],max_seqlen_q=s['mq'],max_seqlen_kv=s['mkv'],mask_mode=4,win_left=s['left'],win_right=0,layout_q='TND',layout_out='TND')
    if op=='GQAPagedFwdOp':
        def run(q,k_pages,v_pages,page_table,cache_seqlens,cu_seqlens_q,q_scale=None,k_scale=None,v_scale=None,rope_cos=None,rope_sin=None,sinks=None):
            assert all(x is None for x in (q_scale,k_scale,v_scale,rope_cos,rope_sin))
            opts=dict(base,cu_seqlens_q=cu_seqlens_q,seqused_kv=cache_seqlens,layout_kv='PA_BBND')
            metadata=flash_attn_metadata(s['h'],s['hkv'],s['d'],**opts)
            call={k:v for k,v in opts.items() if k!='batch_size'}
            output,lse=flash_attn(q,k_pages,v_pages,block_table=page_table,sinks=sinks,attn_mask=mask,metadata=metadata,softmax_scale=s['d']**-0.5,return_softmax_lse=False,**call)
            return output
    else:
        def run(q,k,v,cu_seqlens_q,cu_seqlens_kv,q_scale=None,k_scale=None,v_scale=None,rope_cos=None,rope_sin=None,sinks=None):
            assert all(x is None for x in (q_scale,k_scale,v_scale,rope_cos,rope_sin))
            opts=dict(base,cu_seqlens_q=cu_seqlens_q,cu_seqlens_kv=cu_seqlens_kv,layout_kv='TND')
            metadata=flash_attn_metadata(s['h'],s['hkv'],s['d'],**opts)
            call={k:v for k,v in opts.items() if k!='batch_size'}
            output,lse=flash_attn(q,k,v,sinks=sinks,attn_mask=mask,metadata=metadata,softmax_scale=s['d']**-0.5,return_softmax_lse=False,**call)
            return output
    return run
