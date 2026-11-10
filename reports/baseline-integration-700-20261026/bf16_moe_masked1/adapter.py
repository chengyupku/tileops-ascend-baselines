from pathlib import Path
import importlib.util
from dataclasses import replace

def _load_parent():
 p=Path(__file__).resolve().parent.parent/'bf16_moe_hadamard_parent10'/'adapter.py'
 s=importlib.util.spec_from_file_location('parent_adapter',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def make_binding(native_product): return _load_parent().make_binding(native_product)

def make_candidate(case):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate, source_hashes
 # Reuse the fixed original dense_fp32 API through an explicit Gemm-shaped bridge.
 bp=Path(__file__).resolve().parent.parent/'bf16_local_gemm18'/'binding.py'
 s=importlib.util.spec_from_file_location('orig_binding',bp);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 class Sig: name='GemmFwdOp'
 class Fake:
  _call=type('Call',(),{'signature':Sig()})()
  params={'trans_a':False,'trans_b':False}
 orig=m.dense_fp32(Fake())
 def native(lhs,rhs,metadata=None):
  return orig.implementation.run(lhs,rhs)
 binding=_load_parent().make_binding(native)
 op=case._call.signature.name
 if op=='MoEGroupedGemmFwdOp':
  def run(a,b,metadata):
   p=case.params; out_dtype=getattr(torch,p['out_dtype']) if p.get('out_dtype') else a.dtype
   layout=p['layout']; masked='masked' in layout
   return binding.moe(a,b,metadata,out_dtype=out_dtype,masked=masked)
  desc='Parent-fixed V5 MoE wrapper; original dense_fp32 bridged as lambda(lhs,rhs,metadata), metadata observed only; exact layout/defined-row scatter and requested dtype.'
 elif op=='HadamardTransformFwdOp':
  def run(x): return binding.hadamard(x,n=case.params.get('n'),base_order=case.params.get('base_order',1))
  desc='Parent-fixed V5 Hadamard wrapper; original dense_fp32 bridged for both FP32 stages and exact coefficient/packing contract.'
 else: raise ValueError(op)
 return Candidate(Implementation(run), 'ops-transformer::aclnnGroupedMatmulV5', desc,
  source_hashes(Path(__file__),bp,Path(__file__).resolve().parent.parent/'bf16_moe_hadamard_parent10'/'adapter.py',Path(__file__).resolve().parent.parent/'bf16_local_gemm18'/'binding.py'), cleanup=orig.cleanup)
