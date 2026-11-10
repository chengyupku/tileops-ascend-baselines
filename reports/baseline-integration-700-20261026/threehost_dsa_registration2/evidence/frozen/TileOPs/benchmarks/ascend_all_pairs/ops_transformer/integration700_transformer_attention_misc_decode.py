"""Complete original FlashAttentionScore with layout and cache addressing bindings."""
from pathlib import Path

def factory(case):
    import torch
    from benchmarks.ascend_all_pairs.ops_transformer.validated_runtime import verify_pins,load
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    verify_pins('attention');original=load('ascend_transformer_graph');binding=original.GraphCall('ops-transformer')
    op=case._call.signature.name
    def attention(q,k,v,causal=False):
        b,n,h,d=q.shape
        out=torch.empty((b,n,h,v.shape[-1]),device=q.device,dtype=q.dtype)
        maximum=torch.empty((b,h,n,8),device=q.device,dtype=torch.float32);total=torch.empty_like(maximum)
        unused=torch.empty((0,0,0,0),device=q.device,dtype=q.dtype)
        mask=None
        if causal:
            mask=torch.triu(torch.ones((n,k.shape[1]),device=q.device,dtype=torch.bool),diagonal=k.shape[1]-n+1)
        inputs=[q.contiguous(),k.contiguous(),v.contiguous(),None,None,None,mask,*([None]*12)]
        binding.call('FlashAttentionScore',inputs,[maximum,total,unused,out],{'scale_value':d**-0.5,'keep_prob':1.0,'pre_tockens':2147483647,'next_tockens':2147483647,'head_num':h,'input_layout':'BSND','inner_precise':0,'sparse_mode':0})
        return out
    def run(*args):
        if op=='MLADecodeWithKVCacheFwdOp':
            q,qp,k,kp=args
            return attention(torch.cat((q,qp),-1).unsqueeze(1),torch.cat((k,kp),-1),k).squeeze(1)
        q,k,v,lens,table=args;ps=case.params['page_size'];lens=lens.cpu().tolist();rows=[]
        for b in range(q.shape[0]):
            ids=table[b].long();kk=k.reshape(-1,ps,*k.shape[1:])[ids].reshape(1,-1,*k.shape[1:])[:,:lens[b]]
            vv=v.reshape(-1,ps,*v.shape[1:])[ids].reshape(1,-1,*v.shape[1:])[:,:lens[b]]
            rows.append(attention(q[b:b+1],kk,vv,case.params['is_causal']))
        return torch.cat(rows,0)
    return Candidate(Implementation(run),'ops-transformer::FlashAttentionScore','Full original attention; concatenate MLA Q/K features, or gather all live paged KV and supply causal mask. Return complete output.',source_hashes(Path(__file__),Path(original.__file__)),cleanup=binding.cleanup)
