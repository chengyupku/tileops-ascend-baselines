"""Stateless InstanceNorm through one complete original BatchNormV3 call."""
from .common import candidate, new_binding

def instance_train(case):
    import torch
    assert case.params['use_input_stats'] and case.inputs[1] is None
    binding=new_binding()
    def run(x, running_mean, running_var, weight, bias):
        n,c=x.shape[:2]
        xx=x.contiguous().reshape(1,n*c,1,-1)
        weight=torch.ones(c,dtype=x.dtype,device=x.device) if weight is None else weight
        bias=torch.zeros(c,dtype=x.dtype,device=x.device) if bias is None else bias
        weight=weight.repeat(n).contiguous();bias=bias.repeat(n).contiguous()
        mean=torch.zeros(n*c,dtype=torch.float32,device=x.device);var=torch.ones_like(mean)
        out=torch.empty_like(xx);saved_mean=torch.empty_like(mean);saved_rstd=torch.empty_like(var)
        binding.call('BatchNormV3',[xx,weight,bias,mean,var],[out,mean,var,saved_mean,saved_rstd],
                     {'epsilon':float(case.params['eps']),'momentum':float(case.params['momentum']),'is_training':True},
                     formats=[0,2,2,2,2,0,2,2,2,2])
        return out.reshape(x.shape)
    return candidate(binding,run,'BatchNormV3','Stateless InstanceNorm packs N*C independent channels into [1,N*C,1,spatial], replicates per-channel affine over N, and selects y from one complete training API; canonical input dtype and complete spatial axis retained.', 'norm/batch_norm_v3')
