"""Prepared interface for original ops-nn InstanceNorm; requires its own deployed package."""
def factory(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from benchmarks.ascend_all_pairs.ops_nn.common import new_binding, candidate
    if case._call.signature.name != 'InstanceNormFwdOp' or case.id != 'wider-channel-affine-float16':
        raise UnsupportedCase('Only the no-running-statistics exact contract is admitted')
    x, running_mean, running_var, weight, bias = case.inputs
    if running_mean is not None or running_var is not None or not case.params.get('use_input_stats', True):
        raise UnsupportedCase('Original InstanceNorm API does not consume or update running statistics')
    assert x.dtype == torch.float16 and weight.dtype == bias.dtype == x.dtype
    binding = new_binding()
    eps = float(case.params['eps'])
    def run(x, running_mean, running_var, weight, bias):
        if running_mean is not None or running_var is not None:
            raise UnsupportedCase('Running statistics are outside this original API contract')
        y = torch.empty_like(x)
        stats_shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        mean = torch.empty(stats_shape, dtype=weight.dtype, device=x.device)
        variance = torch.empty_like(mean)
        binding.call('InstanceNorm', [x.contiguous(), weight.contiguous(), bias.contiguous()],
                     [y, mean, variance], {'data_format': 'ND', 'epsilon': eps})
        return y
    return candidate(binding, run, 'InstanceNorm',
                     'One complete original InstanceNorm API; unchanged FP16 operands/affine; original epsilon and full output. No component composition.',
                     'norm/instance_norm')
