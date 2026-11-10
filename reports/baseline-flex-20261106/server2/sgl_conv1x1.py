"""Candidate only: complete 1x1 convolution through original SGL vector LoRA."""
from pathlib import Path
import hashlib

def build(case):
    import importlib
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, LIBRARIES, _owned_native, source_hashes
    p=case.params
    pair=lambda x:(x,x) if isinstance(x,int) else tuple(x)
    x,w,*_=case.inputs
    if tuple(w.shape[2:])!=(1,1) or p['groups']!=1 or pair(p['stride'])!=(1,1) or pair(p['padding'])!=(0,0):
        raise ValueError('Exact pointwise convolution only')
    if x.dtype not in (torch.float16,torch.bfloat16) or w.dtype!=x.dtype or w.shape[0]>3072:
        raise ValueError('Exact dtype and <=3 full-K native calls required')
    root=LIBRARIES/'sgl-kernel-npu';package=root/'python/sgl_kernel_npu/sgl_kernel_npu'
    sys.path.insert(0,str(package.parent));loaded=importlib.import_module('sgl_kernel_npu')
    if Path(loaded.__file__).resolve()!=(package/'__init__.py').resolve():raise RuntimeError('Ownership mismatch')
    binary=package/'lib/libsgl_kernel_npu.so'
    _owned_native(binary,['_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l'])
    def run(x,w,bias=None):
        n,cin,h,width=x.shape;cout=w.shape[0]
        a=x.permute(0,2,3,1).contiguous().reshape(-1,cin);weight=w.reshape(cout,cin)
        if bias is not None:
            a=torch.cat((a,torch.ones((a.shape[0],1),device=a.device,dtype=a.dtype)),1)
            weight=torch.cat((weight,bias[:,None]),1)
        outputs=[]
        for first in range(0,cout,1024):
            last=min(cout,first+1024);cols=last-first;aligned_cols=(cols+15)//16*16
            k=a.shape[1];padded_k=max((k+15)//16*16,aligned_cols+16)
            xx=F.pad(a,(0,padded_k-k));ww=F.pad(weight[first:last],(0,padded_k-k,0,aligned_cols-cols)).unsqueeze(0)
            indices=torch.zeros(1,dtype=torch.int32,device=x.device)
            lengths=torch.tensor([a.shape[0]],dtype=torch.int32,device=x.device)
            ranks=torch.tensor([aligned_cols],dtype=torch.int32,device=x.device)
            scales=torch.ones(1,dtype=torch.float16,device=x.device)
            output=torch.empty((a.shape[0],aligned_cols),dtype=torch.float32,device=x.device)
            torch.ops.npu.sgemmv_shrink(xx,ww,indices,lengths,ranks,scales,output,1)
            outputs.append(output[:,:cols].to(x.dtype))
        return torch.cat(outputs,1).reshape(n,h,width,cout).permute(0,3,1,2).contiguous()
    sources=[f for f in (root/'csrc/lora').rglob('*') if f.suffix in ('.cpp','.h','.hpp')]
    hashes=source_hashes(binary,package/'__init__.py',root/'csrc/pytorch_extensions.cpp',*sources)
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return Candidate(Implementation(run),'SGL original sgemmv_shrink pointwise convolution','Full reduction per output-column group; bias constant coordinate; all packing and metadata inside run.',hashes)
