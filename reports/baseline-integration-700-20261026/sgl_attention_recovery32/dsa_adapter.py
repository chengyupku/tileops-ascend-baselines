"""Pack selected KV as independent requests; original SGL executes all math."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes,UnsupportedCase

def build(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_mla',host)
    p=case.params;tail=p['dim_tail'];stride=p['stride_kv']
    def run(q,kv,indices):
        b,s,h,total=q.shape;sk,g=kv.shape[1:3];d=total-tail;group=h//g
        if g!=1 or stride!=1:raise UnsupportedCase('This exact-scope adapter requires one KV head and stride one.')
        start=p.get('q_start_index_s');start=sk-s if start is None else start
        # Metadata only on CPU: match the reference's set-valued selection and causal limit.
        selected=indices.detach().cpu().tolist()
        out=torch.empty((b,s,h,d),device=q.device,dtype=q.dtype)
        for batch in range(b):
            for begin in range(0,s,64):
                end=min(s,begin+64)
                rows=[sorted(set(j for j in selected[batch][i][0] if 0<=j<sk and j<=start+i)) for i in range(begin,end)]
                if any(not r for r in rows):raise UnsupportedCase('Empty selection requires separate original API validation.')
                n=max(map(len,rows));pages=(n+127)//128;width=pages*128
                ix=torch.tensor([r+[0]*(width-len(r)) for r in rows],device=q.device,dtype=torch.int64)
                data=kv[batch,ix,0]
                kk=data[...,:d].contiguous().reshape(-1,128,1,d)
                kr=F.pad(data[...,d:],(0,64-tail)).contiguous().reshape(-1,128,1,64)
                qq=F.pad(q[batch,begin:end],(0,64-tail)).contiguous()
                lengths=torch.tensor(list(map(len,rows)),device=q.device,dtype=torch.int32)
                table=torch.arange((end-begin)*pages,device=q.device,dtype=torch.int32).reshape(end-begin,pages)
                yy=torch.empty((end-begin,h,d),device=q.device,dtype=q.dtype)
                scale=p.get('sm_scale');scale=total**-0.5 if scale is None else scale
                fn(qq,kk,kr,yy,lengths,scale,128,table)
                out[batch,begin:end]=yy
        return out
    return Candidate(Implementation(run),'sgl_kernel_npu.attention.decode_attention.decode_mla','Exact selected causal KV set packed as independent full requests; page padding and zero rotary tail only; original complete attention API unchanged.',source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),timing_mode='host_roundtrip')
PROVIDERS={'DSADecodeWithKVCacheFwdOp':build}
