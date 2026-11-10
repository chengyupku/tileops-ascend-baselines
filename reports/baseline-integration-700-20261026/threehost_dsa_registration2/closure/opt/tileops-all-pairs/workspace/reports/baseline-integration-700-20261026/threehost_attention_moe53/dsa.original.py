"""Original complete FlashAttentionScore with sparse selection represented as mask metadata."""
from pathlib import Path

def factory(case):
    import torch
    from benchmarks.ascend_all_pairs.ops_transformer.validated_runtime import verify_pins,load
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    verify_pins('attention');original=load('ascend_transformer_graph');binding=original.GraphCall('ops-transformer')
    def run(q,kv,indices):
        b,s,h,dq=q.shape;sk=kv.shape[1];hk=kv.shape[2];dv=dq-case.params['dim_tail']
        ix=indices.cpu().long().permute(0,2,1,3)
        # A boolean selection mask naturally deduplicates repeated physical indices.
        selected=torch.zeros((b,hk,s,sk+1),dtype=torch.bool).scatter_(3,ix,True)[...,:sk]
        if case.params['is_causal']:
            position=torch.arange(s)+case.params['q_start_index_s'];keypos=(torch.arange(sk)+1)*case.params['stride_kv']-1
            selected &= (position[:,None]>=keypos[None,:])[None,None,:,:]
        mask=(~selected).contiguous().to(q.device)
        out=torch.empty((b,s,h,dv),device=q.device,dtype=q.dtype);mx=torch.empty((b,h,s,8),device=q.device,dtype=torch.float32);sm=torch.empty_like(mx);unused=torch.empty((0,0,0,0),device=q.device,dtype=q.dtype)
        scale=case.params.get('sm_scale');scale=dq**-0.5 if scale is None else scale
        binding.call('FlashAttentionScore',[q.contiguous(),kv.contiguous(),kv[...,:dv].contiguous(),None,None,None,mask,*([None]*12)],[mx,sm,unused,out],{'scale_value':float(scale),'keep_prob':1.0,'pre_tockens':2147483647,'next_tockens':2147483647,'head_num':h,'input_layout':'BSND','inner_precise':0,'sparse_mode':0})
        return out
    return Candidate(Implementation(run),'ops-transformer::FlashAttentionScore','Represent selected keys, sentinel and compressed causal bounds as boolean metadata; original full attention and value-prefix selection, no arithmetic outside the complete API.',source_hashes(Path(__file__),Path(original.__file__)),cleanup=binding.cleanup)
