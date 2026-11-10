"""Selected KV addressing followed by one complete original attention call."""
from pathlib import Path

def factory(case):
    import torch
    from benchmarks.ascend_all_pairs.ops_transformer.validated_runtime import verify_pins,load
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    verify_pins('attention');original=load('ascend_transformer_graph');binding=original.GraphCall('ops-transformer')
    def run(q,k,v,indices,counts,offsets,token_indices):
        t,h,d=q.shape;hk=k.shape[1];bs=case.params['block_size'];sel=indices.shape[-1];width=bs*sel;b=t*hk
        # Integer metadata translation only: retain duplicate selected slots and mask tails.
        ix=indices.cpu().long();ct=counts.cpu().long();off=offsets.cpu().long();tokens=token_indices.cpu().long()
        local=(ix[...,None]*bs+torch.arange(bs)).reshape(t,hk,width)
        start=off[tokens[:,0]][:,None,None];length=(off[1:]-off[:-1])[tokens[:,0]][:,None,None]
        valid=(torch.arange(width)[None,None,:]//bs<ct[...,None]) & (local<length)
        if case.params['is_causal']:valid &= local<=tokens[:,1,None,None]
        addresses=(start+local).clamp(0,t-1).to(q.device)
        head=torch.arange(hk,device=q.device)[None,:,None]
        kk=k[addresses,head].reshape(b,width,1,d);vv=v[addresses,head].reshape(b,width,1,d)
        qq=q.reshape(t,hk,h//hk,d).reshape(b,1,h//hk,d).contiguous();out=torch.empty_like(qq)
        maximum=torch.empty((b,h//hk,1,8),device=q.device,dtype=torch.float32);total=torch.empty_like(maximum);unused=torch.empty((0,0,0,0),device=q.device,dtype=q.dtype)
        mask=(~valid).reshape(b,1,1,width).to(q.device)
        binding.call('FlashAttentionScore',[qq,kk,vv,None,None,None,mask,*([None]*12)],[maximum,total,unused,out],{'scale_value':float(case.params['scale']),'keep_prob':1.0,'pre_tockens':2147483647,'next_tockens':2147483647,'head_num':h//hk,'input_layout':'BSND','inner_precise':2,'sparse_mode':0})
        return out.reshape(t,h,d)
    return Candidate(Implementation(run),'ops-transformer::FlashAttentionScore','Only selected KV address packing and full explicit mask; complete native attention output with manifest scale and all-mask mode.',source_hashes(Path(__file__),Path(original.__file__)),cleanup=binding.cleanup)
