"""v2: original ChunkKdaFwd, FP16 QKV preserved; exact FP32 gate/beta promotion.
No replacement arithmetic or normalization. Device correctness remains untested.
"""
from adapter import _candidate, recurrent_family

def dispatch(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from native_call import NativeCall as GraphCall
    p=case.params; op=case._call.signature.name
    if case.inputs[0].dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase('ChunkKdaFwd QKV supports FP16/BF16 only')
    if any(p.get(k) for k in ('use_qk_l2norm_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
        raise UnsupportedCase('ChunkKdaFwd has no QK epsilon normalization or beta sigmoid/negative-eigenvalue flag; no replacement arithmetic')
    binding=GraphCall('ops-transformer')
    def run(*inputs):
        decode=op=='DeltaNetRecurrentFwdOp'
        if decode:
            q,k,v,beta,initial=inputs
            q,k,v=[x[:,None] for x in (q,k,v)]; beta=beta[:,None]
            g=offsets=a_log=dt_bias=None; scale=1.
        elif op=='DeltaNetFwdOp':
            q,k,v,beta,initial,offsets,_=inputs;g=a_log=dt_bias=None
            scale=p.get('scale') or q.shape[-1]**-.5
        else:
            q,k,v,g,beta,initial,offsets,_,a_log,dt_bias=inputs
            scale=p.get('scale') or q.shape[-1]**-.5
        if p.get('use_gate_in_kernel'):raise UnsupportedCase('Neutral padding requires already transformed gates')
        batch,length,heads,dk=q.shape; hv,dv=v.shape[-2:]
        spans=[(i,0,length) for i in range(batch)] if offsets is None else [(0,a,b) for a,b in zip(offsets.tolist()[:-1],offsets.tolist()[1:])]
        state_shape=(len(spans),hv,dv,dk) if p.get('state_v_first') else (len(spans),hv,dk,dv)
        states=torch.zeros(state_shape,dtype=torch.float32,device=q.device) if initial is None else initial.float().contiguous().clone()
        output=torch.empty_like(v)
        for sequence,(row,start,stop) in enumerate(spans):
            if start==stop: continue
            qs,ks,vs=[x[row:row+1,start:stop].contiguous() for x in (q,k,v)]
            gate=torch.zeros((1,stop-start,hv,dk),dtype=torch.float32,device=q.device) if g is None else g[row:row+1,start:stop].float()
            bias=dt_bias
            if gate.ndim==3:
                gate=gate[...,None].expand(-1,-1,-1,dk)
                if bias is not None: bias=bias[:,None].expand(-1,dk).reshape(-1)
            padded_length=((stop-start+63)//64)*64
            pad=padded_length-(stop-start)
            bs=beta[row:row+1,start:stop].float().contiguous()
            if pad:
                qs,ks,vs=[torch.nn.functional.pad(x,(0,0,0,0,0,pad)) for x in (qs,ks,vs)]
                gate=torch.nn.functional.pad(gate,(0,0,0,0,0,pad))
                bs=torch.nn.functional.pad(bs,(0,0,0,pad))
            state=states[sequence:sequence+1]
            if p.get('state_v_first'): state=state.transpose(-1,-2)
            state=state.contiguous()
            out=torch.empty_like(vs); final=torch.empty_like(state)
            optional_float=torch.empty((1,),dtype=torch.float32,device=q.device)
            optional_data=[torch.empty((1,),dtype=q.dtype,device=q.device) for _ in range(6)]
            aqk=torch.empty((1,hv,padded_length,64),dtype=q.dtype,device=q.device);akk=torch.empty_like(aqk)
            binding.call('ChunkKdaFwd',[qs,ks,vs,gate.contiguous(),bs,a_log,None if bias is None else bias.float().contiguous(),state,None,None],
                [out,final,optional_float,aqk,akk,*optional_data],
                {'layout':'BSND','scale':float(scale),'chunk_size':64,'safe_gate':p.get('lower_bound') is not None,'lower_bound':float(p.get('lower_bound') or -5.),'use_gate_in_kernel':bool(p.get('use_gate_in_kernel',False)),'state_v_first':False})
            output[row:row+1,start:stop].copy_(out[:,:stop-start]);states[sequence:sequence+1].copy_(final.transpose(-1,-2) if p.get('state_v_first') else final)
        return (output[:,0],states.to(initial.dtype)) if decode else (output,states)
    candidate=_candidate(binding,run,'ChunkKdaFwd','attention/chunk_kda_fwd','v2 FP16 native QKV; lossless FP32 gate/beta promotion; scalar gate broadcasting; per-sequence state views; unchanged original checker',host=True)
    return candidate
