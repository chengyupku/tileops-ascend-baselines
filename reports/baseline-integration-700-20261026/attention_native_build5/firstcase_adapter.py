"""UNEXECUTED candidate for exactly GQAPagedFwdOp/gpt-oss-sink-decode-bfloat16-bfloat16.
Import/call only in supervisor-authorized npu-guard reservation after deployment.
Single-query causal attention sees every valid cached key; mask_mode=0 is equivalent.
All original output/oracle checks must still run. This is not a registered provider.
"""
def firstcase(q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q,
              q_scale=None, k_scale=None, v_scale=None, rope_cos=None,
              rope_sin=None, sinks=None):
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
    assert tuple(q.shape) == (8, 64, 64)
    assert tuple(k_pages.shape) == tuple(v_pages.shape) == (256, 64, 8, 64)
    assert sinks is not None and tuple(sinks.shape) == (64,)
    assert all(x is None for x in (q_scale, k_scale, v_scale, rope_cos, rope_sin))
    # Canonical exactkey supplies eight one-token requests and 2048 valid KV tokens each.
    options = dict(cu_seqlens_q=cu_seqlens_q, seqused_kv=cache_seqlens,
                   max_seqlen_q=1, max_seqlen_kv=2048, mask_mode=0,
                   win_left=-1, win_right=-1, layout_q='TND',
                   layout_kv='PA_BBND', layout_out='TND')
    metadata = flash_attn_metadata(64, 8, 64, batch_size=8, **options)
    output, unused_lse = flash_attn(q, k_pages, v_pages, block_table=page_table,
                                  sinks=sinks, metadata=metadata, softmax_scale=0.125,
                                  return_softmax_lse=False, **options)
    return output
