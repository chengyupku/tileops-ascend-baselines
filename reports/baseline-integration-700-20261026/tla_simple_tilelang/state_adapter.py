import hashlib,json
from pathlib import Path

def factory(case,op,native,tl,out):
 import torch
 import torch.nn.functional as F
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 p=case.params
 if op=='CausalConv1dFwdOp':
  src='examples/causal_conv1d/causal_conv1d.py';name='get_causal_conv1d_fn_pipeline_v15'
  m=native.load(src,name);kernel=m.get_causal_conv1d_fn_pipeline_v15(4,2,64,'float32')
  if kernel.adapter.libpath is None:kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
  def run(x,weight,conv_state,cu_seqlens,bias=None,has_initial_state=None):
   assert bias is None and p['activation']=='silu'
   offsets=cu_seqlens.cpu().tolist();flags=has_initial_state.cpu().tolist() if has_initial_state is not None else [False]*(len(offsets)-1)
   y=torch.empty_like(x);state=torch.empty_like(conv_state)
   for bi in range(0,len(offsets)-1,2):
    lengths=[offsets[j+1]-offsets[j]+32 for j in [bi,bi+1]]
    assert all(v%96==0 for v in lengths)
    cs=torch.zeros((2,3,64),dtype=torch.float32,device=x.device)
    cu=torch.tensor([0,lengths[0],sum(lengths)],dtype=torch.int32,device=x.device)
    idx=torch.arange(2,dtype=torch.int32,device=x.device);im=torch.zeros_like(idx)
    for di in range(0,x.shape[0],64):
     packed=[]
     for j in [bi,bi+1]:
      prefix=torch.zeros((32,64),dtype=torch.float32,device=x.device)
      if flags[j]:prefix[-3:]=conv_state[j,di:di+64].T.float()
      packed.append(torch.cat([prefix,x[di:di+64,offsets[j]:offsets[j+1]].T.float()],0))
     result=kernel(torch.cat(packed,0).contiguous(),weight[di:di+64].T.float().contiguous(),cs,cu,idx,im)
     start=0
     for ji,j in enumerate([bi,bi+1]):
      y[di:di+64,offsets[j]:offsets[j+1]]=result[start+32:start+lengths[ji]].T.to(x.dtype)
      state[j,di:di+64]=cs[ji].T.to(x.dtype)
      start+=lengths[ji]
   return y,state
 elif op=='MoEGroupedGemmFwdOp':
  src='examples/gemm/example_gemm_pto_developer.py';name='matmul'
  m=native.load(src,name)
  kernels={}
  def run(a,b,metadata):
   offsets=[0]+metadata.cpu().tolist();N=b.shape[1];K=b.shape[2];np=(N+255)//256*256
   result=torch.empty((a.shape[0],N),device=a.device,dtype=torch.float32)
   for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
    if start==end:continue
    mm=end-start;mp=(mm+127)//128*128
    if mp not in kernels:
     kernels[mp]=m.matmul(mp,np,K,128,256,64,dtype='float')
     if kernels[mp].adapter.libpath is None:kernels[mp].adapter.lib_generator.libpath=kernels[mp].adapter.lib._name
    result[start:end]=kernels[mp](F.pad(a[start:end].float(),(0,0,0,mp-mm)),F.pad(b[i].T.contiguous().float(),(0,np-N)))[:mm,:N]
   return result
 else:raise ValueError(op)
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [native.VENDOR/src,Path(__file__)]}
 (out/'native.json').write_text(json.dumps({'api':src+':'+name,'params':p,'hashes':hashes},indent=2))
 return Candidate(Implementation(tilelang_run(run,hashes,out/'jit')),'tilelang-ascend::'+src+':'+name,'Complete original API; public independent channel/expert slices, input/state packing.',hashes)
