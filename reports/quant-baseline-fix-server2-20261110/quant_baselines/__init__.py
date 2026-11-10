"""Quantization baseline variants. Import/run under the host NPU guard.

Original workload references and checkers remain external and unchanged.
Only exact cases listed in the accompanying verified registry are measured
candidates. A factory alone is not a correctness or performance certificate.
"""
def make(op, params, reference, variant):
    from .layouts import factory
    if variant == 'torch-eager-control':
        if op == 'INT4QuantPerGroupFwdOp':
            from .layouts import int4_group
            return lambda w: int4_group(w, params['group_size'])
        return reference
    if variant == 'torch-eager-layout-fixed':
        if op not in {'INT8QuantPerBlockFwdOp','FP8QuantPerBlockFwdOp','INT8DequantPerBlockFwdOp'}:
            raise ValueError('No layout correction for this family')
        return factory(op,params,reference)
    if variant == 'ops-math-native-broadcast':
        from .ops_math import factory as ops_factory
        fn, _ = ops_factory(op, params, params['q'][0])
        return fn
    if variant == 'torch-npu-antiquant':
        from .native import factory as native_factory
        fn=native_factory(op,params)
        if fn is None:raise ValueError('Original native API does not support this output contract')
        return fn
    if variant in ('torch-compile-inductor-ascendc','torch-compile-inductor-antiquant'):
        import os,torch
        if os.environ.get('TORCHINDUCTOR_NPU_BACKEND')!='ascendc':
            raise RuntimeError('Select the verified AscendC backend before importing torch_npu')
        torch._dynamo.config.suppress_errors=False
        fn=factory(op,params,reference)
        if variant=='torch-compile-inductor-antiquant':
            from .native import factory as native_factory
            fn=native_factory(op,params)
            if fn is None:raise ValueError('Native antiquant contract unavailable')
        return torch.compile(fn,backend='inductor',fullgraph=True,dynamic=False)
    raise ValueError(variant)
