"""Prepare exact workload frequency tables once; rotation stays input-dependent."""
from adapters.execution_boundary import ExecutionBoundary


def bind(workload, args, precision_context=None):
    import contextlib
    from workloads import rope
    x = args[0]
    name = workload.call.signature.name
    params = workload.call.params
    context = precision_context() if precision_context else contextlib.nullcontext()
    with context:
        if name == 'RoPENeoxPositionIdsFwdOp':
            cos, sin = rope.rope_frequency_tables(
                params.get('rotary_dim') or x.shape[-1], params['max_position'],
                base=params['base'], dtype=x.dtype, device=x.device)
            def execute(x, position_ids):
                return rope.ref_rope_neox_position_ids(
                    x, cos, sin, position_ids, params.get('rotary_dim'))
        else:
            tables = {'RoPEFwdOp': rope.rope_frequency_tables,
                      'RoPELlama31FwdOp': rope.llama31_frequency_tables,
                      'YaRNFwdOp': rope.yarn_frequency_tables,
                      'LongRoPEFwdOp': rope.longrope_frequency_tables}[name]
            seq_len = x.shape[0] if params['input_layout'] == '1d' else x.shape[1]
            kwargs = {k:v for k,v in params.items() if k not in ('input_layout','rope_layout')}
            if 'rescale_factors' in workload.call.specs:
                # Use the actual workload materialization, never regenerate factors.
                kwargs['rescale_factors'] = workload.tensors['rescale_factors']
            cos, sin = tables(x.shape[-1], seq_len, dtype=x.dtype, device=x.device, **kwargs)
            rotate = rope.ref_rope_non_neox if params.get('rope_layout','neox') == 'interleaved' else rope.ref_rope_neox
            def execute(x):
                return rotate(x, cos, sin)
    return ExecutionBoundary(execute, tuple(args), lambda:None,
        {'operator':name,'tables_outside_steady_state':True,
         'runtime_position_gather_inside':name=='RoPENeoxPositionIdsFwdOp',
         'table_dtype':str(cos.dtype),'table_shape':list(cos.shape),
         'table_device':str(cos.device),'cold_setup_measured_separately':False})
