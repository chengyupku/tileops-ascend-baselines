"""Delta training backward with forward graph prepared outside measurement.

Saved state tensors are independently checked by the case's correctness contract;
this graph is recreated per case, never reused across input values.
"""
from adapters.execution_boundary import ExecutionBoundary

def bind(args, chunk_size):
    import torch
    from adapters.delta_training_eager import reference as forward
    do,q,k,v,beta,*saved=args
    with torch.enable_grad():
        variables=tuple(x.detach().float().requires_grad_(True) for x in (q,k,v,beta))
        output=forward(*variables,chunk_size)[0]
        grad_output=do.float()
    def execute(upstream=grad_output):
        with torch.enable_grad():
            gradients=torch.autograd.grad(output,variables,grad_outputs=upstream,
                                           retain_graph=True,create_graph=False)
        return tuple(x.to(q.dtype) for x in gradients)
    return ExecutionBoundary(execute,(grad_output,),lambda:None,
        {'operator':'DeltaNetChunkBwdOp','forward_outside_timing':True,
         'backward_recomputed_each_window':True,'output_cast_inside_timing':True,
         'requires_saved_state_contract_check':True})
