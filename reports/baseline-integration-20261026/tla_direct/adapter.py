"""Complete upstream APIs with only public dtype/layout/shape adaptation."""
import hashlib,json
from pathlib import Path

def factory(case,op,native,tl,out):
 import torch
 import torch.nn.functional as F
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 p=case.params
 if op=='CausalConv1dUpdateFwdOp':
  src='examples/causal_conv1d/causal_conv1d_decode.py';api='_build_decode_kernel_jit'
  m=native.load(src,'build_causal_conv1d_decode_kernel','pass_configs_config');batch,dim=p['x'][0];padded=(dim+2047)//2048*2048
  kernel=tl.compile(m.build_causal_conv1d_decode_kernel(4,padded//2048,dtype_str=p['x'][1],has_silu=p.get('activation')=='silu'),out_idx=[-1],pass_configs=m.pass_configs_config,target='pto')
  api='build_causal_conv1d_decode_kernel'
  def run(x,weight,conv_state,bias=None):
   xx=F.pad(x.contiguous(),(0,padded-dim));ww=F.pad(weight.T.contiguous(),(0,padded-dim))
   state=F.pad(conv_state[...,1:].transpose(1,2).contiguous(),(0,padded-dim)).clone()
   idx=torch.arange(batch,device=x.device,dtype=torch.int32);flags=torch.ones_like(idx)
   bb=torch.zeros(dim,dtype=x.dtype,device=x.device) if bias is None else bias
   y=kernel(xx,ww,state,idx,idx,flags,F.pad(bb,(0,padded-dim)))[:,:dim].contiguous()
   result_state=torch.cat((conv_state[...,1:2],state[...,:dim].transpose(1,2)),-1)
   return y,result_state
 elif op=='RMSNormFwdOp':
  src='examples/normalization/rms_norm.py';api='_rms_norm';m=native.load(src,api)
  shape=p['x'][0];hidden=shape[-1];rows=1
  for d in shape[:-1]:rows*=d
  eps=2**-23 if p.get('eps') is None else p['eps']
  kernel=tl.compile(m._rms_norm.__wrapped__(rows,hidden,64,64,eps,dtype='float'),out_idx=[1],pass_configs=m.pass_configs,target='pto')
  def run(x,weight=None,bias=None):
   assert weight is None and bias is None
   return kernel(x.contiguous().reshape(rows,hidden).float()).reshape(shape).to(x.dtype)
 elif op=='WhereFwdOp':
  src='testing/python/language/test_tilelang_ascend_language_select.py';api='select_kernel_mod2'
  m=native.load(src,api,'PASS_CONFIGS','BLOCK_M','BLOCK_N')
  # Original float16 test configuration, original supported PTO target.
  shape=p['output'][0];rows,cols=shape;dt=p['output'][1];assert dt=='float16'
  kernel=tl.compile(m.select_kernel_mod2(rows,cols,m.BLOCK_M[dt],m.BLOCK_N[dt],dt),out_idx=[-1],pass_configs=m.PASS_CONFIGS,target='pto')
  def run(mask,yes,no):
   bits=mask.contiguous().reshape(rows,cols//8,8).to(torch.uint8)
   packed=torch.zeros((rows,cols//8),dtype=torch.uint8,device=mask.device)
   for bit in range(8):packed.bitwise_or_(bits[...,bit]<<bit)
   return kernel(yes.contiguous(),no.contiguous(),packed)
 else:raise ValueError(op)
 paths=[native.VENDOR/src,Path(__file__),Path(__file__).with_name('run_case.py')]
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 (out/'native.json').write_text(json.dumps({'api':src+':'+api,'params':p,'hashes':hashes,'runtime':tl.__file__,'library':str(tl._LIB_PATH)},indent=2))
 calls=[]
 def checked_launch(*args):
  result=run(*args);torch.npu.synchronize();calls.append({'returned':True,'synchronized':True})
  (out/'launch.json').write_text(json.dumps(calls,indent=2));return result
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 return Candidate(Implementation(tilelang_run(checked_launch,hashes,out/'jit')),'tilelang-ascend::'+src+':'+api,'Original full API and upstream settings; public layout/dtype/state packaging only.',hashes)
