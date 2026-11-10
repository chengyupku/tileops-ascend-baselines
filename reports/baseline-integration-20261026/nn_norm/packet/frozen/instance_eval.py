"""Complete original BatchNormV3 inference preserving canonical input dtypes."""
from .common import candidate, new_binding

def instance_eval(case):
    import torch
    assert not case.params['use_input_stats'] and case.inputs[1] is not None
    binding=new_binding()
    def run(x, running_mean, running_var, weight, bias):
        channels=x.shape[1]
        xx=x.contiguous().reshape(x.shape[0],channels,1,-1)
        weight=torch.ones(channels,dtype=x.dtype,device=x.device) if weight is None else weight.contiguous()
        bias=torch.zeros(channels,dtype=x.dtype,device=x.device) if bias is None else bias.contiguous()
        mean=running_mean.contiguous().clone();var=running_var.contiguous().clone()
        out=torch.empty_like(xx);saved_mean=torch.empty_like(mean);saved_rstd=torch.empty_like(var)
        binding.call('BatchNormV3',[xx,weight,bias,mean,var],[out,mean,var,saved_mean,saved_rstd],
                     {'epsilon':float(case.params['eps']),'momentum':float(case.params['momentum']),'is_training':False},
                     formats=[0,2,2,2,2,0,2,2,2,2])
        return out.reshape(x.shape),mean,var
    return candidate(binding,run,'BatchNormV3','Inference uses unchanged canonical FP16 input/affine and FP32 running statistics, supported in original ascend950 registration; preserve complete output and running statistics.', 'norm/batch_norm_v3')
