import hashlib,json
from pathlib import Path

def factory(case,op,native,tl,out):
 import torch
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 p=case.params
 if op=='TopKSelectFwdOp':
  src='examples/topk_selector/example_topk_selector.py';name='simple_topk_selector'
  m=native.load(src,name)
  kernel=native.compile_builder(m,name,1,16384,2048,512,out_idx=[-1])
  def run(x,starts,ends):
   assert bool((starts==0).all()) and bool((ends==16384).all())
   return kernel(x.reshape(1,16384).contiguous()).reshape(1,1,1,2048)
 elif op=='GemmINT8W8A8FwdOp':
  src='examples/deepseek_v4/int8_gemm.py';name='int8_gemm_kernel_corrected'
  m=native.load(src,name)
  a,b=case.inputs[:2];N=b.shape[0];K=a.shape[1]
  kernel=native.compile_builder(m,name,N,K,out_dtype='bfloat16')
  def run(a,b,sa,sb):return kernel(a.contiguous(),b.contiguous(),sa.reshape(-1).contiguous(),sb.reshape(-1).contiguous())
 elif op=='AllFwdOp':
  src='testing/python/language/test_tilelang_ascend_language_elementwise.py';name='reduce_runtime_semantics_kernel'
  x=case.inputs[0];assert x.dtype==torch.bool, 'Float truth conversion adds non-original math'
  assert p.get('dim',-1)==-1
  cols=x.shape[-1];rows=x.numel()//cols
  m=native.load(src,name,'pass_configs')
  kernel=tl.compile(m.reduce_runtime_semantics_kernel(1,cols,'min',-1,dtype='float'),out_idx=[-1],pass_configs=m.pass_configs,target='ascendc')
  def run(x):
   values=x.reshape(rows,cols).float()
   return torch.cat([kernel(values[i:i+1]) for i in range(rows)]).reshape(p['output'][0]).bool()
 else:raise ValueError(op)
 if kernel.adapter.libpath is None:kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [native.VENDOR/src,Path(__file__)]}
 (out/'native.json').write_text(json.dumps({'api':src+':'+name,'params':p,'hashes':hashes},indent=2))
 return Candidate(Implementation(tilelang_run(run,hashes,out/'jit')),'tilelang-ascend::'+src+':'+name,'Complete original API with public shape and metadata binding only.',hashes)
