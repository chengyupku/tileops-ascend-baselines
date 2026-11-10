"""Padding-safe batched mean pooling, with all live metadata inside timing."""
import torch
from adapters.execution_boundary import ExecutionBoundary

def bind(case,args,previous):
    if case._call.signature.name!='MeanPoolingFwdOp':
        from candidates_v4 import bind as other
        return other(case,args,previous)
    chunk=case._call.params['chunk_size'];dtype=case.workload.accum_dtype
    def run(x,*metadata):
        if not metadata or metadata[0] is None:
            b,s,h,d=x.shape
            if s%chunk:raise ValueError('uniform tail needs explicit bounds')
            return x.reshape(b,s//chunk,chunk,h,d).mean(2,dtype=dtype).to(x.dtype)
        offsets,indices=metadata
        starts=offsets[indices[:,0].long()].long()+indices[:,1].long()*chunk
        ends=offsets[indices[:,0].long()+1].long()
        loc=starts[:,None]+torch.arange(chunk,device=x.device)
        valid=loc<ends[:,None]
        values=x[:,loc.clamp_max(x.shape[1]-1)]
        masked=torch.where(valid[None,:,:,None,None],values,0.)
        sums=masked.sum(2,dtype=dtype)
        return (sums/valid.sum(1)[None,:,None,None]).to(x.dtype)
    return ExecutionBoundary(run,tuple(args),lambda:None,{'implementation':'torch-npu eager','candidate':'v6','input_dependent_work_timed':True,'change':'batch chunk means; ragged gather and bounds stay timed; mask excludes NaN in padding','targets':[1,4]})
