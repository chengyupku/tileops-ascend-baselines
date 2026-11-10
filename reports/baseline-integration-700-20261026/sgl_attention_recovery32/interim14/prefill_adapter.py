from pathlib import Path
from attention_page16 import paged
def prefill_append(case):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase,LIBRARIES,local_function,source_hashes
    from cache_adapter import paged_cache_write
    q,kn,vn,kp,vp,ks,vs,cu,oldlens,table=case.inputs
    if kp.dtype!=q.dtype or vp.dtype!=q.dtype:
        raise UnsupportedCase('Original store/decode API cannot provide the declared FP8 cache quantization/dequantization scales.')
    page=case.params['page_size'];h=kn.shape[-2];d=kn.shape[-1]
    p={**case.params,'pos_encoding_mode':'none'}
    native=paged(SimpleNamespace(inputs=(q,),params=p))
    host='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    rotate=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',host)
    from workloads.rope import rope_frequency_tables
    store=paged_cache_write(SimpleNamespace(inputs=(kn,vn,kp.reshape(-1,page,h,d),vp.reshape(-1,page,h,d),torch.empty(q.shape[0],device=q.device,dtype=torch.int64),None,None),params={}))
    def run(q,kn,vn,kp,vp,ks,vs,cu,oldlens,table):
        bounds,old=cu.cpu().tolist(),oldlens.cpu().tolist()
        pos=torch.cat([torch.arange(n,n+bounds[i+1]-bounds[i],device=q.device) for i,n in enumerate(old)])
        slots=torch.cat([table[i,pos[bounds[i]:bounds[i+1]]//page].long()*page+pos[bounds[i]:bounds[i+1]]%page for i in range(len(old))])
        if case.params.get('fuse_rope'):
            rd=case.params.get('rotary_dim') or d
            maxpos=max(n+bounds[i+1]-bounds[i] for i,n in enumerate(old))
            cos,sin=rope_frequency_tables(rd,maxpos,base=case.params['rope_base'],dtype=q.dtype,device=q.device)
            cache=torch.cat((cos,sin),dim=-1)[pos].float().contiguous()
            q,_=rotate(q.float().contiguous(),q.float().contiguous(),cache,rd,True)
            kn,_=rotate(kn.float().contiguous(),kn.float().contiguous(),cache,rd,True)
            q,kn=q.to(vn.dtype),kn.to(vn.dtype)
        kout,vout=store.implementation.run(kn,vn,kp.reshape(-1,page,h,d),vp.reshape(-1,page,h,d),slots)
        lengths=oldlens+(cu[1:]-cu[:-1]).to(oldlens.dtype)
        output=native.implementation.run(q,kout,vout,table,lengths,cu)
        return output,kout.reshape_as(kp),vout.reshape_as(vp)
    return replace(native,implementation=Implementation(run),api=native.api+' + original prefix-valid KV store + original RoPE',
        adaptation='Use actual sequence metadata to place new keys/values into private original pools with native store; optional native rotate new rows only; native paged attention reads updated lengths. Return attention and both complete caches.',
        source_sha256=native.source_sha256|store.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host,Path(__file__).parents[3]/'TileOPs/workloads/rope.py'))

PROVIDERS={'GQAPrefillPagedWithKVCacheFwdOp':prefill_append}
