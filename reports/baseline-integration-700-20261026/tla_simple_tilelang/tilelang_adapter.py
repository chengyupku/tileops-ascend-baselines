import json,hashlib,importlib.util,sys,math
from pathlib import Path
W=Path('/opt/tileops-all-pairs/workspace');V=W/'tilelang'
def load(src):
 spec=importlib.util.spec_from_file_location('_integration700_original_'+Path(src).stem,V/src);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def factory(case,op,out):
 import torch,tilelang
 import torch.nn.functional as F
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 p=case.params
 if op=='FP8QuantFwdOp':
  src='examples/ascend/example_per_token_cast_to_fp8.py';api='per_token_cast_simt';m=load(src)
  x=case.inputs[0];hidden=x.shape[-1];rows=x.numel()//hidden;assert rows%32==0 and hidden in [64,128]
  kernel=m.per_token_cast_simt(rows,128)
  def run(x):
   q,s=kernel(F.pad(x.float().reshape(rows,hidden),(0,128-hidden)))
   return s.reshape(x.shape[:-1]),q[:,:hidden].contiguous().reshape(x.shape)

 elif op in {'BmmFwdOp','MoEGroupedGemmFwdOp'}:
  src='examples/ascend/example_gemm.py';api='gemm';m=load(src)
  kernels={}
  def one(a,b):
   mm,kk=a.shape;nn=b.shape[0];mp=(mm+255)//256*256;np=(nn+255)//256*256
   dt='bfloat16' if a.dtype==torch.bfloat16 else 'float32'
   key=(mp,kk,np,dt)
   if key not in kernels:kernels[key]=tilelang.compile(m.gemm(mp,kk,np,dtype=dt,out_dtype=dt,hf32=None),target='ascend',out_idx=-1)
   return kernels[key](F.pad(a.to(getattr(torch,dt)),(0,0,0,mp-mm)),F.pad(b.to(getattr(torch,dt)),(0,0,0,np-nn)))[:mm,:nn].to(a.dtype)
  kernel=None
  if op=='BmmFwdOp':
   def run(a,b):return torch.stack([one(a[i].contiguous(),b[i].T.contiguous()) for i in range(a.shape[0])])
  else:
   def run(a,b,metadata):
    offsets=[0]+metadata.cpu().tolist();result=torch.empty((a.shape[0],b.shape[1]),dtype=a.dtype,device=a.device)
    for i,(start,end) in enumerate(zip(offsets[:-1],offsets[1:])):
     if start<end:result[start:end]=one(a[start:end].contiguous(),b[i].contiguous())
    return result
 elif op=='GQADenseFwdOp':
  src='examples/ascend/flash_attention/core.py';api='flash_attention_fwd';m=load(src)
  q,k,v=case.inputs[:3];B,SQ,H,D=q.shape;HK=k.shape[2];SK=k.shape[1];G=H//HK
  assert SQ==1 and H%HK==0
  kernel=tilelang.compile(m.flash_attention_fwd(128,SK,128,softmax_scale=1/math.sqrt(D)),target='ascend',out_idx=-1)
  def run(q,k,v):
   result=torch.empty_like(q)
   for bi in range(B):
    for hi in range(HK):
     qq=F.pad(q[bi,0,hi*G:(hi+1)*G].contiguous(),(0,128-D,0,128-G))
     kk=F.pad(k[bi,:,hi].contiguous(),(0,128-D));vv=F.pad(v[bi,:,hi].contiguous(),(0,128-D))
     result[bi,0,hi*G:(hi+1)*G]=kernel(qq,kk,vv)[:G,:D]
   return result
 elif op=='MHCPostFwdOp':
  src='examples/deepseek_mhc/example_mhc_post.py';api='mhc_post_tilelang';m=load(src)
  x,h,res=case.inputs;B,D=x.shape;HC=h.shape[-1]
  eye=torch.eye(HC,dtype=torch.float32,device=x.device).unsqueeze(0).expand(B,HC,HC).contiguous()
  out_tensor=torch.empty((B,HC,D),dtype=x.dtype,device=x.device)
  prim=m.mhc_post_tilelang.get_tir(eye,res.reshape(B,HC,D),h,x,out_tensor,HC,D)
  kernel=tilelang.compile(prim,target='ascend',pass_configs=m.mhc_post_tilelang.pass_configs)
  def run(x,h,res):
   output=torch.empty((B,HC,D),dtype=x.dtype,device=x.device)
   kernel(eye,res.reshape(B,HC,D),h,x,output)
   return output.reshape(B,HC*D)
 else:raise ValueError(op)
 paths=[V/src,Path(__file__)]
 runtime=Path(tilelang.__file__).parent
 paths+=list(runtime.rglob('*.py'))+list((runtime/'lib').glob('*.so'))
 hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
 (out/'native.json').write_text(json.dumps(dict(api=src+':'+api,params=p,hashes=hashes,runtime=str(runtime)),indent=2))
 calls=[]
 def wrapped(*args):
  result=run(*args);torch.npu.synchronize()
  calls.append(dict(returned=True,kernel_type=str(type(kernel)),adapter_type=str(type(kernel.adapter)) if kernel is not None else 'per-shape original GEMM'))
  (out/'launch.json').write_text(json.dumps(calls,indent=2))
  return result
 return Candidate(Implementation(wrapped),'tilelang::'+src+':'+api,'Original complete SIMT FP8 quantizer; public zero padding, FP32 widening and output reshape only.',hashes)
