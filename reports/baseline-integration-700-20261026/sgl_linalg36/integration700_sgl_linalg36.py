"""Original SGL full-K public SGEMMV with independent output column packing."""
from pathlib import Path
import os, hashlib
SOURCE = Path('/opt/tileops-all-pairs/build/sgl-pristine-20261025T2310/source')
def build(case):
    import torch, importlib, sys
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes, _owned_native
    package=SOURCE/'python/sgl_kernel_npu/sgl_kernel_npu'
    sys.path.insert(0,str(package.parent))
    mod=importlib.import_module('sgl_kernel_npu')
    assert Path(mod.__file__).resolve()==(package/'__init__.py').resolve()
    binary=package/'lib/libsgl_kernel_npu.so'
    _owned_native(binary,['_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l'])
    def mm(a,w):
        m,k=a.shape; n=w.shape[0]
        out=torch.empty((m,n),dtype=torch.float32,device=a.device)
        indices=torch.zeros(1,dtype=torch.int32,device=a.device)
        lengths=torch.tensor([m],dtype=torch.int32,device=a.device)
        scales=torch.ones(1,dtype=torch.float16,device=a.device)
        for c in range(0,n,64):
            width=min(64,n-c); padded_n=(width+15)//16*16
            padded_k=max((k+15)//16*16,padded_n+16)
            x=F.pad(a,(0,padded_k-k)).contiguous()
            weight=F.pad(w[c:c+width],(0,padded_k-k,0,padded_n-width)).contiguous().unsqueeze(0)
            ranks=torch.tensor([padded_n],dtype=torch.int32,device=a.device)
            y=torch.empty((m,padded_n),dtype=torch.float32,device=a.device)
            torch.ops.npu.sgemmv_shrink(x,weight,indices,lengths,ranks,scales,y,1)
            torch.npu.synchronize()
            out[:,c:c+width].copy_(y[:,:width])
        return out.to(a.dtype)
    op=case._call.signature.name
    def run(*args):
        a,b=args[:2]
        if op=='GemmFwdOp':
            return mm(a.T if case.params['trans_a'] else a,b if case.params['trans_b'] else b.T)
        if op=='BmmFwdOp':
            return torch.stack([mm(a[i],b[i].T) for i in range(a.shape[0])])
        sizes,offsets=args[2:]; ta=case.params['transpose_a'];tb=case.params['transpose_b']; outputs=[]
        for i,(begin,length) in enumerate(zip(offsets.cpu().tolist(),sizes.cpu().tolist())):
            aa=a[begin:begin+length]
            bb=(b[:,begin:begin+length] if tb else b[begin:begin+length]) if ta else b[i]
            if length==0:
                assert not ta, 'No synthetic reduction for empty K'
                outputs.append(torch.empty((0,bb.shape[0] if tb else bb.shape[1]),device=a.device,dtype=a.dtype));continue
            outputs.append(mm(aa.T if ta else aa,bb if tb else bb.T))
        return torch.stack(outputs) if ta else torch.cat(outputs,dim=0)
    files=[p for p in (SOURCE/'csrc/lora').rglob('*') if p.suffix in ('.cpp','.h','.hpp')]
    return Candidate(Implementation(run),'sgl_kernel_npu::sgemmv_shrink full-K FP32 output (private linalg36)',
      'Original public API, scale half 1; independent output columns64; aligned zero padding; complete K; transpose/group packing; output cast; synchronous lifetime.',
      {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),binary,package/'__init__.py',SOURCE/'csrc/pytorch_extensions.cpp',*files]},timing_mode='host_roundtrip')
