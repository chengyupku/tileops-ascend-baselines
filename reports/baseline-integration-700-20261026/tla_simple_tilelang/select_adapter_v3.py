import hashlib,json
from pathlib import Path

def factory(case,op,native,tl,out):
 import torch
 from benchmarks.ascend_baselines import Candidate
 from benchmarks.api import Implementation
 from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
 p=case.params
 src='testing/python/language/test_tilelang_ascend_language_select.py'
 m=native.load(src,'select_kernel_mod2','PASS_CONFIGS','BLOCK_M','BLOCK_N')
 shape=p['output'][0];rows,cols=shape
 dt='float32'
 kernel=tl.compile(m.select_kernel_mod2(rows,cols,m.BLOCK_M[dt],m.BLOCK_N[dt],dt),out_idx=[-1],pass_configs=m.PASS_CONFIGS,target='pto')
 # Cached runtime omits libpath; bind metadata to the actual loaded CDLL.
 if kernel.adapter.libpath is None:
  kernel.adapter.lib_generator.libpath=kernel.adapter.lib._name
 def run(*inputs):
  if op=='WhereFwdOp': mask,yes,no=inputs
  else:
   no,mask,*other=inputs
   yes=other[0] if other else torch.full((),p['value'],dtype=no.dtype,device=no.device)
  bits=mask.expand(shape).contiguous().reshape(rows,cols//8,8).to(torch.uint8)
  packed=torch.zeros((rows,cols//8),dtype=torch.uint8,device=mask.device)
  for bit in range(8):packed.bitwise_or_(bits[...,bit]<<bit)
  result=kernel(yes.expand(shape).contiguous().float(),no.expand(shape).contiguous().float(),packed)
  return result.to(getattr(torch,p['output'][1]))
 paths=[native.VENDOR/src,Path(__file__),Path(__file__).with_name('run_case.py')]
 hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
 (out/'native.json').write_text(json.dumps({'api':src+':select_kernel_mod2','params':p,'hashes':hashes,'runtime':tl.__file__},indent=2))
 return Candidate(Implementation(tilelang_run(run,hashes,out/'jit')),'tilelang-ascend::'+src+':select_kernel_mod2','Original complete select; upstream float32 tiles, lossless widening and mask packing.',hashes)
