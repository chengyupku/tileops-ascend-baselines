import json
from pathlib import Path
import torch,torch_npu,tilelang
import tilelang.ascend.language as T
from workloads.elementwise import sinusoidal_reference
r=Path(__file__).resolve().parent
N=65536
@T.prim_func
def main(A:T.Tensor((N,),'float32'),B:T.Tensor((N,),'float32'),Q:T.Tensor((N,),'float32'),S:T.Tensor((N,),'float32'),C:T.Tensor((N,),'float32')):
 with T.Kernel(N//1024) as bx:
  a=T.alloc_shared((1024,),'float32');b=T.alloc_shared((1024,),'float32');q=T.alloc_shared((1024,),'float32');s=T.alloc_shared((1024,),'float32');c=T.alloc_shared((1024,),'float32')
  T.copy(A[bx*1024:(bx+1)*1024],a);T.copy(B[bx*1024:(bx+1)*1024],b)
  with T.SimdVF():
   mask=T.simd.pset(32)
   for i in range(16):
    av=T.simd.vld(a[i*64]);bv=T.simd.vld(b[i*64]);v=T.simd.vdiv(av,bv,mask,precision='exact');T.simd.vsts(q[i*64],v,mask)
  with T.SimtVF(threads=1024):
   for i in T.Parallel(1024):
    s[i]=T.sin(q[i]);c[i]=T.cos(q[i])
  T.copy(q,Q[bx*1024:(bx+1)*1024]);T.copy(s,S[bx*1024:(bx+1)*1024]);T.copy(c,C[bx*1024:(bx+1)*1024])
k=tilelang.compile(main,target='ascend',out_idx=[2,3,4]);pos=torch.arange(512,dtype=torch.float32)[:,None].expand(512,128).contiguous();dim=torch.arange(0,256,2,dtype=torch.float32);freq=torch.pow(10000.,dim/256);den=freq[None].expand(512,128).contiguous();q,s,c=[v.cpu().reshape(512,128) for v in k(pos.flatten().npu(),den.flatten().npu())];refq=pos/den;got=torch.stack([s,c],-1).reshape(512,256);cpu=sinusoidal_reference(512,256,torch.float32,'cpu');npu=sinusoidal_reference(512,256,torch.float32,'npu').cpu()
def diff(a,b):return dict(max_abs=float((a-b).abs().max()),mismatch=int(((a-b).abs()>1e-5+1e-5*b.abs()).sum()))
report=dict(exact_angle_cpu=diff(q,refq),angle_bit_mismatch=int((q.view(torch.int32)!=refq.view(torch.int32)).sum()),exact_kernel_cpu=diff(got,cpu),exact_kernel_npu=diff(got,npu),cpu_npu=diff(cpu,npu),runtime_limit=torch.npu.get_device_limit(0));print(json.dumps(report,indent=2));(r/'sinusoidal-angle-probe.json').write_text(json.dumps(report,indent=2)+'\n');(r/'sinusoidal-angle-probe.c').write_text(k.get_kernel_source());torch.npu.synchronize()
