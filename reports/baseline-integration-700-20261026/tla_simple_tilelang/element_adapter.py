import hashlib,json
from pathlib import Path
from functools import partial

MAP={'AbsFwdOp':('vec_abs',128,256),'BitwiseNotFwdOp':('bitwise_not',128,256),'LogicalNotFwdOp':('bitwise_xor',128,256),'ClampScalarFwdOp':('clamp',64,64),'ExpFwdOp':('exp',128,256),'LogFwdOp':('ln',128,256),'NegFwdOp':('vec_muls',64,32),'SinFwdOp':('sin',128,128),'CosFwdOp':('cos',128,128),'SqrtFwdOp':('sqrt',128,256),'RsqrtFwdOp':('rsqrt',128,256),'ReciprocalFwdOp':('reciprocal',128,128),'IsnanFwdOp':('compare',128,256)}

def factory(case,op,native,tl,out):
 import torch
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 p=case.params;shape,odt=p['output'];rows,cols=shape
 src='testing/python/language/test_tilelang_ascend_language_elementwise.py'
 sources=[src]
 if op=='RoundFwdOp':
  src='testing/python/base/unary_op.py';name='make_unary_kernel'
  m=native.load(src,name);c=native.load('testing/python/base/common.py','DEFAULT_PASS_CONFIGS')
  kernel=tl.compile(m.make_unary_kernel(partial(tl.language.tile.round,count=2048))(rows,cols,64,64,'float'),out_idx=[-1],pass_configs=c.DEFAULT_PASS_CONFIGS,target='ascendc')
  sources=[src,'testing/python/base/common.py']
 else:
  name,bm,bn=MAP[op];m=native.load(src,name,'pass_configs')
  dt='int16' if op in {'LogicalNotFwdOp','BitwiseNotFwdOp'} else 'float'
  if op=='BitwiseNotFwdOp': cols*=2
  if op=='BitwiseNotFwdOp': assert odt=='int32'
  args=[]
  if op=='ClampScalarFwdOp':args=[p['max'],p['min']]
  if op=='NegFwdOp':args=[-1]
  if op=='IsnanFwdOp':args=['NE']
  kernel=tl.compile(getattr(m,name)(rows,cols,bm,bn,*args,dtype=dt),out_idx=[-1],pass_configs=m.pass_configs,target='ascendc')
 if kernel.adapter.libpath is None: kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
 def run(x):
  v=x.contiguous().to(torch.int16 if op=='LogicalNotFwdOp' else (torch.float32))
  if op=='BitwiseNotFwdOp':v=x.contiguous().view(torch.int16)
  args=[v]
  if op=='LogicalNotFwdOp':args.append(torch.ones_like(v))
  if op=='IsnanFwdOp':args.append(v)
  y=kernel(*args)
  if op=='IsnanFwdOp':
   shifts=torch.arange(8,dtype=torch.uint8,device=y.device)
   y=((y.reshape(-1,1)>>shifts)&1).reshape(shape)
  if op=='BitwiseNotFwdOp':return y.contiguous().view(torch.int32)
  return y.to(getattr(torch,odt))
 paths=[native.VENDOR/s for s in sources]+[Path(__file__)]
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 (out/'native.json').write_text(json.dumps({'api':src+':'+name,'params':p,'hashes':hashes},indent=2))
 return Candidate(Implementation(tilelang_run(run,hashes,out/'jit')),'tilelang-ascend::'+src+':'+name,'Original full API; upstream tile settings; public lossless dtype/constant bindings.',hashes)
