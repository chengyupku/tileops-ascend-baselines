"""Same normalization reference formulas on caller-owned private running stats."""
from adapters.execution_boundary import private_inputs

def bind(case,args):
    import torch
    import torch.nn.functional as F
    p=case._call.params;name=case._call.signature.name
    eps=p.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
    def run(x,rm,rv,weight,bias):
        if name=='BatchNormFwdOp':
            xf=x.float();y=F.batch_norm(xf,rm,rv,None if weight is None else weight.float(),None if bias is None else bias.float(),training=p['training'],momentum=p['momentum'],eps=eps)
            if not p['training']:return y.to(x.dtype),None,None,rm,rv
            var,mean=torch.var_mean(xf,dim=[0,*range(2,x.ndim)],correction=0)
            return y.to(x.dtype),mean,torch.rsqrt(var+eps),rm,rv
        return F.instance_norm(x.float(),rm,rv,None if weight is None else weight.float(),None if bias is None else bias.float(),p['use_input_stats'],p['momentum'],eps).to(x.dtype)
    indices=tuple(i for i in (1,2) if args[i] is not None)
    return private_inputs(run,args,indices)
