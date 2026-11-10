"""Explicit routes for corrected measurement, rejecting unimplemented timing fixes."""
from adapters.execution_boundary import ExecutionBoundary, private_inputs
from adapters.protection_reference import remove_protection_clones

def bind(case, args, current_fn, reasons):
    name=case._call.signature.name
    if name in ('MLADenseBwdOp','MLAVarlenBwdOp'):
        from adapters.mla_backward_saved import reference
        w=case.workload
        def run(*xs):return reference(*xs,is_causal=w.is_causal,sm_scale=w.sm_scale)
        return ExecutionBoundary(run,tuple(args),lambda:None,{'route':'mla_supplied_lse_backward','uses_supplied_lse':True,'probability_recomputation_timed':True,'forward_lse_reduction_excluded':True,'all_gradient_computation_timed':True})
    if 'frequency_table_cache_boundary' in reasons:
        from adapters.rope_cached import bind as rope_bind
        from adapters.rope_pow_v10 import RopePower
        return rope_bind(case.workload,args,precision_context=RopePower)
    if 'input_protection_timing' in reasons:
        if name=='GQAPrefillPagedWithKVCacheFwdOp':
            fn,proof=remove_protection_clones(case.reference,['k_pages','v_pages'])
            result=private_inputs(fn,args,(3,4));result.description['source_transform']=proof
            return result
        if name in ('MLAKVCacheWriteFwdOp','PagedKVCacheWriteFwdOp'):
            names,indices=(['kv_cache'],(2,)) if name=='MLAKVCacheWriteFwdOp' else (['k_cache','v_cache'],(2,3))
            fn,proof=remove_protection_clones(case.reference,names)
            result=private_inputs(fn,args,indices);result.description['source_transform']=proof
            return result
        if name in ('BatchNormFwdOp','InstanceNormFwdOp'):
            from adapters.norm_private import bind as norm_bind
            return norm_bind(case,args)
        if name=='FusedQKNormRoPEFwdOp':
            from workloads.norm import fused_qk_norm_rope_heads
            w=case.workload
            def run(qkv,q_weight,k_weight,cos_sin_cache,positions):
                qkv[:,: (w.num_heads+w.num_kv_heads)*w.dim]=fused_qk_norm_rope_heads(
                    qkv,q_weight,k_weight,cos_sin_cache,positions,w.num_heads,w.num_kv_heads,w.eps,w.rope_layout)
                return qkv
            return private_inputs(run,args,(0,))
        if name in ('SSDRecurrentFwdOp','CausalConv1dFwdOp','CausalConv1dUpdateFwdOp','PagedKVCacheGatherFwdOp'):
            index={'SSDRecurrentFwdOp':5,'CausalConv1dFwdOp':2,'CausalConv1dUpdateFwdOp':2,'PagedKVCacheGatherFwdOp':0}[name]
            return private_inputs(current_fn,args,(index,) if args[index] is not None else ())
        raise NotImplementedError('protection route still requires audited adapter: '+name)
    if 'forward_recomputation_in_backward' in reasons:
        if name in ('GQADenseBwdOp','GQABwdOp'):
            from adapters.backward_only_v2 import bind_gqa
            return bind_gqa(case.workload,args)
        if name=='EngramGateConvBwdOp':
            from adapters.backward_only_v2 import bind_engram
            return bind_engram(case.workload,args)
        if name=='GLAChunkBwdOp':
            from adapters.backward_only_v2 import bind_gla
            return bind_gla(case.workload,args)
        if name=='DeltaNetChunkBwdOp':
            from adapters.delta_backward_only_v2 import bind as delta_bind
            return delta_bind(args,case._call.params['chunk_size'])
        raise NotImplementedError('backward-only route must be implemented before measurement: '+name)
    return ExecutionBoundary(current_fn,tuple(args),lambda:None,{'no_known_boundary_correction':True})
