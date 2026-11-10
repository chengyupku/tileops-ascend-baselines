"""Thin complete original ACLNN APIs; no metadata or attention algorithm."""
import ctypes as C
import json,os,ast
from pathlib import Path
P=C.c_void_p; I=C.c_int64; U=C.c_uint64; IP=C.POINTER(I)
R=Path(__file__).resolve().parent
EVENTS=[]
def guard():
 m=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
 assert m['owner']=='baseline-single-server2/nsa-compressed' and m['phase']=='running' and m['child']['pid']==os.getpgrp()
def bind(lib,name,args,result=C.c_int32):
 f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
def check(rc,name):
 EVENTS.append(dict(stage=name,returncode=rc))
 if rc: raise RuntimeError(f'{name}: {rc}')
class API:
 def __init__(self):
  guard()
  import torch
  self.t=torch
  base=C.CDLL('/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libnnopbase.so')
  vendor=Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/attention_generated_recovery5/native_install/vendors/attention_native_build5_transformer')
  self.lib=C.CDLL(str(vendor/'op_api/lib/libcust_opapi.so'))
  self.create=bind(base,'aclCreateTensor',[IP,U,C.c_int,IP,I,C.c_int,IP,U,P],P)
  self.destroy=bind(base,'aclDestroyTensor',[P])
  self.getmeta=bind(self.lib,'aclnnFlashAttnMetadataGetWorkspaceSize',[P]*4+[I]*10+[C.c_char_p]*3+[C.c_bool,P,C.POINTER(U),C.POINTER(P)])
  self.getattn=bind(self.lib,'aclnnFlashAttnGetWorkspaceSize',[P]*11+[C.c_double]+[I]*5+[C.c_char_p]*3+[I,P,P,C.POINTER(U),C.POINTER(P)])
 def call(self,name,inputs,attrs,outputs):
  guard();keep=[];descs=[]
  def tensor(t):
   if t is None:return None
   assert t.device.type in ('npu','privateuseone')
   import torch_npu
   fmt=torch_npu.get_npu_format(t)
   assert fmt in (0,1,2,30), f'Non-base storage format unsupported: {fmt}'
   aclfmt={3:47,4:0,5:30}.get(t.ndim,2)
   EVENTS.append(dict(descriptor_shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),storage_bytes=t.untyped_storage().nbytes(),torch_format=fmt,acl_format=aclfmt))
   dt={self.t.float32:0,self.t.float16:1,self.t.bfloat16:27,self.t.int32:3,self.t.int64:9,self.t.int8:2}
   dims=(I*t.ndim)(*t.shape);stride=(I*t.ndim)(*t.stride())
   storage=(I*1)(t.untyped_storage().nbytes()//t.element_size())
   d=self.create(dims,t.ndim,dt[t.dtype],stride,t.storage_offset(),aclfmt,storage,1,t.untyped_storage().data_ptr())
   if not d:raise RuntimeError('aclCreateTensor null')
   descs.append(d);keep.extend([t,dims,stride,storage]);return d
  try:
   ins=list(map(tensor,inputs));outs=list(map(tensor,outputs));size=U();executor=P()
   get=self.getmeta if name=='FlashAttnMetadata' else self.getattn
   check(get(*ins,*attrs,*outs,C.byref(size),C.byref(executor)),name+'GetWorkspaceSize')
   assert executor.value
   workspace=self.t.empty((size.value,),dtype=self.t.uint8,device=outputs[0].device) if size.value else None
   keep.append(workspace)
   stream=self.t.npu.current_stream().npu_stream
   check(bind(self.lib,'aclnn'+name,[P,U,P,P])(workspace.data_ptr() if workspace is not None else None,size.value,executor,stream),name)
   self.t.npu.synchronize()
   EVENTS.append(dict(api=name,workspace_bytes=size.value,returned_and_synchronized=True))
  finally:
   self.t.npu.synchronize()
   for d in descs:check(self.destroy(d),'aclDestroyTensor')
_API=None
def api():
 global _API
 if _API is None:_API=API()
 return _API

def flash_attn_metadata(num_heads_q,num_heads_kv,head_dim,head_dim_v=None,cu_seqlens_q=None,cu_seqlens_kv=None,seqused_q=None,seqused_kv=None,batch_size=-1,max_seqlen_q=-1,max_seqlen_kv=-1,mask_mode=0,win_left=-1,win_right=-1,layout_q='BSND',layout_kv='BSND',layout_out='BSND',is_grad_enabled=False):
 a=api()
 # Execute exact original allocation helpers by AST, excluding builder/import/JIT side effects.
 source=Path('/opt/tileops-all-pairs/workspace/.cache/all-pairs/root_server2_20261022/source_templates/ascend_baselines/ops-transformer/attention/flash_attn/torch_extension/flash_attn.py')
 tree=ast.parse(source.read_text());names={'_get_npu_core_nums','_calculate_batch_size','_fag_deter_elems','_calculate_metadata_size'}
 helpers=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
 env={'torch':a.t,'METADATA_STRIDE':16,'FAG_METADATA_SIZE':412};exec(compile(helpers,str(source),'exec'),env)
 b=env['_calculate_batch_size'](batch_size,cu_seqlens_q,seqused_q)
 size=env['_calculate_metadata_size'](b,num_heads_kv,num_heads_q,max_seqlen_q,max_seqlen_kv)
 out=a.t.empty((size,),dtype=a.t.int32,device='npu')
 attrs=[batch_size,max_seqlen_q,max_seqlen_kv,num_heads_q,num_heads_kv,head_dim,-1 if head_dim_v is None else head_dim_v,mask_mode,win_left,win_right,*[x.encode() for x in (layout_q,layout_kv,layout_out)],is_grad_enabled]
 a.call('FlashAttnMetadata',[cu_seqlens_q,cu_seqlens_kv,seqused_q,seqused_kv],attrs,[out]);return out

def flash_attn(q,k,v,block_table=None,cu_seqlens_q=None,cu_seqlens_kv=None,seqused_q=None,seqused_kv=None,sinks=None,attn_mask=None,metadata=None,softmax_scale=1.,mask_mode=0,win_left=-1,win_right=-1,max_seqlen_q=-1,max_seqlen_kv=-1,layout_q='BSND',layout_kv='BSND',layout_out='BSND',return_softmax_lse=False):
 a=api();assert layout_q==layout_out=='TND' and layout_kv in ('PA_BBND','TND')
 assert q.ndim==3 and k.ndim==v.ndim==(4 if layout_kv=='PA_BBND' else 3)
 assert 0 not in k.stride() and 0 not in v.stride()
 out=a.t.empty((q.shape[0],q.shape[1],v.shape[-1]),dtype=q.dtype,device=q.device)
 lse=a.t.empty((q.shape[1],q.shape[0]) if return_softmax_lse else (0,),dtype=a.t.float32,device=q.device)
 attrs=[softmax_scale,mask_mode,win_left,win_right,max_seqlen_q,max_seqlen_kv,*[x.encode() for x in (layout_q,layout_kv,layout_out)],int(return_softmax_lse)]
 a.call('FlashAttn',[q,k,v,block_table,cu_seqlens_q,cu_seqlens_kv,seqused_q,seqused_kv,sinks,attn_mask,metadata],attrs,[out,lse]);return out,lse
