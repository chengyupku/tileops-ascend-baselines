import hashlib,json
from pathlib import Path

def factory(case,op,native,tl,out):
 import torch
 import torch.nn.functional as F
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 p=case.params;a,b=case.inputs;ta,tb=p.get('trans_a',False),p.get('trans_b',False)
 ash=list(a.shape);bsh=list(b.shape)
 if ta:ash[-2:]=ash[-2:][::-1]
 if tb:bsh[-2:]=bsh[-2:][::-1]
 m,k,n=ash[-2],ash[-1],bsh[-1];mp=(m+127)//128*128;np=(n+255)//256*256
 assert k%64==0
 src,name=('examples/batch_gemm/batch_gemm.py','batch_matmul') if op=='BmmFwdOp' else ('examples/gemm/example_gemm.py','matmul')
 module=native.load(src,name)
 args=[ash[0],mp,np,k,128,256,64] if op=='BmmFwdOp' else [mp,np,k,128,256,64]
 kernel=native.compile_builder(module,name,*args,dtype=str(a.dtype).split('.')[-1],out_idx=[-1])
 if kernel.adapter.libpath is None:kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
 def run(a,b):
  a=(a.transpose(-1,-2) if ta else a).contiguous()
  b=(b.transpose(-1,-2) if tb else b).contiguous()
  return kernel(F.pad(a,(0,0,0,mp-m)),F.pad(b,(0,np-n)))[...,:m,:n].contiguous()
 paths=[native.VENDOR/src,Path(__file__)]
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 (out/'native.json').write_text(json.dumps({'api':src+':'+name,'params':p,'hashes':hashes},indent=2))
 return Candidate(Implementation(tilelang_run(run,hashes,out/'jit')),'tilelang-ascend::'+src+':'+name,'Original full API; unchanged upstream 128x256x64; public padding/transposes only.',hashes)
