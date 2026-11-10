"""Original full attention APIs; public tensor/metadata binding only."""
import hashlib,json
from pathlib import Path

def compile_original(native,tl,module,name,*args,**kwargs):
 builder=getattr(module,name);raw=getattr(builder,'__wrapped__',builder);impl=builder.__jit_impl__
 return tl.compile(raw(*args,**kwargs),out_idx=impl.out_idx,workspace_idx=impl.workspace_idx,
                   target="pto",pass_configs=impl.pass_configs,platform="A5",
                   execution_backend=impl.execution_backend,target_host=impl.target_host,compile_flags=impl.compile_flags)

def factory(case,op,native,tl,out):
 import torch
 import torch.nn.functional as F
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 p=case.params; kernels={}
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

 elif op=='SinFwdOp':
  src='testing/python/language/test_tilelang_ascend_language_elementwise.py';api='sin'
  m=native.load(src,api,'pass_configs');shape,odt=p['output'];rows,cols=shape
  kernel=tl.compile(m.sin(rows,cols,128,128,dtype='float'),out_idx=[-1],pass_configs=m.pass_configs,target='ascendc')
  if kernel.adapter.libpath is None:kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
  def run(x):return kernel(x.contiguous().float()).to(getattr(torch,odt))
 elif op=='MLADecodeWithKVCacheFwdOp':
  src='examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py';api='sparse_attention_fwd'
  m=native.load(src,api);batch,heads,dim=p['q'][0];tail=p['q_pe'][0][-1];length,groups=p['k'][0][1:3]
  kernel=compile_original(native,tl,m,api,heads,dim,tail,length,1,kv_group=groups,dtype=p['q'][1])
  def run(q,q_pe,k,k_pe):
   query=torch.cat((q,q_pe),-1).unsqueeze(1).contiguous();key=torch.cat((k,k_pe),-1).contiguous()
   indices=torch.arange(length,dtype=torch.int32,device=q.device).reshape(1,1,1,length).expand(batch,1,groups,length).contiguous()
   return kernel(query,key,indices)[:,0].contiguous()
 else:
  dim=p['q'][0][-1]
  direct=dim>128
  legacy=__import__('os').environ.get('INTEGRATION700_TLA_ROUTE')=='legacy'
  src='examples/flash_attention/flash_attn_bhsd.py' if direct else 'examples/attention_sink/example_gqa_sink_fwd_varlen/example_gqa_sink_fwd_varlen.py'
  api='flash_attention_fwd' if direct else 'gqa_sink_fwd'
  if legacy:src='examples/gqa_fwd_varlen/gqa_fwd_varlen.py';api='flashattn'
  m=native.load(src,api) if direct else native.load(src,api,'build_attention_mask' if legacy else 'make_attention_mask')
  m.DataType=tl.DataType
  if not direct:
   from tilelang.intrinsics import make_zn_layout,make_nz_layout
   m.make_zn_layout=make_zn_layout;m.make_nz_layout=make_nz_layout
  causal=p.get('is_causal',False)
  def one(q,k,v):
   b,sq,h,d=q.shape;sk=k.shape[1];kh=k.shape[2]
   if legacy:
    assert d==128
    mq=(sq+127)//128*128;nk=(sk+127)//128*128;key=(b,mq,nk,h,kh,d)
    if key not in kernels:kernels[key]=compile_original(native,tl,m,api,b,h//kh,h,d,mq,nk,causal)
    qp=F.pad(q.transpose(1,2).contiguous(),(0,0,0,mq-sq));kp=F.pad(k.transpose(1,2).contiguous(),(0,0,0,nk-sk));vp=F.pad(v.transpose(1,2).contiguous(),(0,0,0,nk-sk))
    cq=torch.arange(b+1,dtype=torch.int32,device=q.device)*sq;ck=torch.arange(b+1,dtype=torch.int32,device=q.device)*sk
    mask=m.build_attention_mask(cq,ck,mq,nk,causal,q.device)
    left,right=p.get('window_size_left',-1),p.get('window_size_right',-1)
    qi=torch.arange(mq,device=q.device)[:,None]+sk-sq;ki=torch.arange(nk,device=q.device)[None,:]
    if left>=0:mask=mask*(ki>=qi-left)
    if right>=0:mask=mask*(ki<=qi+right)
    return kernels[key](qp,kp,vp,mask)[:,:,:sq].transpose(1,2).contiguous()
   if direct:
    if sq!=1:raise ValueError('Direct full noncausal API binding requires single query; query-wise prefix replication not implemented')
    assert sk%64==0
    key=(b,sk,h,d)
    if key not in kernels:kernels[key]=compile_original(native,tl,m,api,b,sk,h,d)
    qp=F.pad(q.transpose(1,2).contiguous(),(0,0,0,sk-1))
    kp=k.repeat_interleave(h//kh,dim=2).transpose(1,2).contiguous();vp=v.repeat_interleave(h//kh,dim=2).transpose(1,2).contiguous()
    return kernels[key](qp,kp,vp)[:,:,:1].transpose(1,2).contiguous()
   mq=(sq+63)//64*64;nk=(sk+127)//128*128
   ql=torch.full((b,),sq,dtype=torch.int32,device=q.device);kl=torch.full((b,),sk,dtype=torch.int32,device=q.device)
   mask,nt=m.make_attention_mask(b,mq,nk,ql,kl,causal,64,128,q.device)
   # Public mask tensor represents optional sliding windows without changing QK/softmax/PV.
   left,right=p.get('window_size_left',-1),p.get('window_size_right',-1)
   if left>=0 or right>=0:
    qi=torch.arange(mq,device=q.device)[:,None]+sk-sq;ki=torch.arange(nk,device=q.device)[None,:];invalid=torch.zeros((mq,nk),device=q.device,dtype=torch.bool)
    if left>=0:invalid|=ki<qi-left
    if right>=0:invalid|=ki>qi+right
    full=mask.reshape(b,mq//64,nk//128,64,128).permute(0,1,3,2,4).reshape(b,mq,nk)
    full.masked_fill_(invalid,float('-inf'));mask=full.reshape(b,mq//64,64,nk//128,128).permute(0,1,3,2,4).contiguous().reshape(nt,64,128)
   key=(b,mq,nk,h,kh,d)
   if key not in kernels:kernels[key]=compile_original(native,tl,m,api,b,h//kh,h,128,mq,nk,causal,mask_tiles=nt,sm_scale=d**-.5)
   qp=F.pad(q.transpose(1,2).contiguous(),(0,128-d,0,mq-sq));kp=F.pad(k.transpose(1,2).contiguous(),(0,128-d,0,nk-sk));vp=F.pad(v.transpose(1,2).contiguous(),(0,128-d,0,nk-sk))
   sinks=torch.full((h,),float('-inf'),dtype=q.dtype,device=q.device)
   return kernels[key](qp,kp,vp,sinks,ql,kl,nk,mask)[:,:,:sq,:d].transpose(1,2).contiguous()
  def gather(pages,table,length):
   ps=pages.shape[1];return pages[table[:(length+ps-1)//ps].long()].reshape(-1,*pages.shape[2:])[:length].unsqueeze(0).contiguous()
  if op=='GQADenseFwdOp':
   def run(q,k,v,*optional):
    assert all(x is None for x in optional);return one(q,k,v)
  elif op=='GQAVarlenFwdOp':
   def run(q,k,v,cq,ck,*optional):
    assert all(x is None for x in optional);qs,ks=cq.tolist(),ck.tolist()
    return torch.cat([one(q[a:b][None],k[c:d][None],v[c:d][None])[0] for a,b,c,d in zip(qs[:-1],qs[1:],ks[:-1],ks[1:])],0)
  elif op=='GQAPagedFwdOp':
   def run(q,k,v,table,lens,cq,*optional):
    assert all(x is None for x in optional);qs,ls=cq.tolist(),lens.tolist()
    return torch.cat([one(q[a:b][None],gather(k,table[i],ls[i]),gather(v,table[i],ls[i]))[0] for i,(a,b) in enumerate(zip(qs[:-1],qs[1:]))],0)
  else:
   def run(q,k,v,lens,table):
    pk=k.reshape(-1,p['page_size'],*k.shape[1:]);pv=v.reshape(-1,p['page_size'],*v.shape[1:])
    return torch.cat([one(q[i:i+1],gather(pk,table[i],ln),gather(pv,table[i],ln)) for i,ln in enumerate(lens.tolist())],0)
 paths=[native.VENDOR/src,Path(__file__)];hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 (out/'native.json').write_text(json.dumps(dict(api=src+':'+api,params=p,hashes=hashes,runtime=tl.__file__),indent=2))
 def checked(*args):
  (out/'CALL_STARTED.json').write_text(json.dumps({'phase':'native_call','time':__import__('time').time()}));print('ADAPTER_CALL_STARTED',flush=True)
  torch.save(tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in args),out/'inputs.pt');result=run(*args);print('ADAPTER_CALL_RETURNED',flush=True);torch.npu.synchronize();(out/'CALL_RETURNED.json').write_text(json.dumps({'phase':'synchronized','time':__import__('time').time()}));return result
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 return Candidate(Implementation(tilelang_run(checked,hashes,out/'jit')),'tilelang-ascend::'+src+':'+api,'Original defaults; only public layout, padding and mask metadata binding.',hashes)
