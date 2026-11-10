"""Per-case forward graphs outside timing, fresh backward on every invocation."""
from adapters.execution_boundary import ExecutionBoundary

def bind_gqa(workload,args):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel,SDPBackend
    grad,q,k,v,o,lse=args
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        leaves=tuple(t.detach().float().requires_grad_(True) for t in (q,k,v))
        result=F.scaled_dot_product_attention(*(t.transpose(1,2) for t in leaves),
                    is_causal=workload.is_causal,enable_gqa=True).transpose(1,2)
        upstream=grad.float()
    def execute():
        with torch.enable_grad(),torch.autograd.set_multithreading_enabled(False):
            gradients=torch.autograd.grad(result,leaves,upstream,retain_graph=True)
        return tuple(g.to(q.dtype) for g in gradients)
    return ExecutionBoundary(execute,(),lambda:None,{'forward_outside_timing':True,
        'fresh_backward':True,'output_cast_timed':True,'route':'gqa_math_sdpa_saved_graph'})

def bind_engram(workload,args):
    import torch
    import torch.nn.functional as F
    from workloads.sequence_modeling.engram import _rmsnorm
    dY,H,k,v,wh,wv,cw,*saved=args
    with torch.enable_grad():
        leaves=tuple(t.float().detach().requires_grad_(True) for t in (H,k,v,wh,wv,cw))
        ha,ka,va,wha,wva,cwa=leaves
        hn,_=_rmsnorm(ha,wha,workload.eps);kn,_=_rmsnorm(ka,wha,workload.eps)
        alpha=torch.sigmoid((hn*kn).sum(-1,keepdim=True)/(H.shape[-1]**.5))
        vh=alpha*va;vn,_=_rmsnorm(vh,wva,workload.eps)
        padded=F.pad(vn.permute(0,2,1),(cw.shape[0]-1,0))
        conv=F.conv1d(padded,cwa.T.unsqueeze(1),groups=H.shape[-1]).permute(0,2,1)
        output=F.silu(conv)+vh;upstream=dY.float()
    def execute():
        with torch.enable_grad(),torch.autograd.set_multithreading_enabled(False):
            g=torch.autograd.grad(output,leaves,upstream,retain_graph=True)
        return tuple(t.to(H.dtype) if i<3 else t for i,t in enumerate(g))
    return ExecutionBoundary(execute,(),lambda:None,{'forward_outside_timing':True,
        'fresh_backward':True,'output_cast_timed':True,'route':'engram_saved_graph'})

def bind_gla(workload,args):
    import torch
    from workloads.linear_attention.gla import gla_fwd_chunked_torch
    do,dfinal,q,k,v,g,h=args;p=workload.arguments()
    with torch.enable_grad():
        leaves=[x.float().detach().requires_grad_(True) for x in (q,k,v,g)]
        initial=h[:,0].float().detach()
        if p['has_initial_state']:
            initial.requires_grad_(True);leaves.append(initial)
        out,final=gla_fwd_chunked_torch(*leaves[:4],p['chunk_size'],scale=p['scale'],initial_state=initial)
        upstream=(do.float(),dfinal.float())
    def execute():
        with torch.enable_grad(),torch.autograd.set_multithreading_enabled(False):
            grads=torch.autograd.grad((out,final),leaves,upstream,retain_graph=True)
        return (*(x.float() for x in grads), *(() if p['has_initial_state'] else (None,)))
    return ExecutionBoundary(execute,(),lambda:None,{'forward_outside_timing':True,
        'fresh_backward':True,'initial_state_gradient':p['has_initial_state'],'route':'gla_saved_graph'})
