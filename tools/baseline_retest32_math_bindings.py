"""Top-level original ops-math API binding; no native implementation changes."""
from functools import partial
from types import SimpleNamespace
from pathlib import Path

def add(case):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,source_hashes,UnsupportedCase
 from benchmarks.ascend_all_pairs.ops_math.graph import make
 p=case.params
 if p.get('alpha',1)!=1:raise UnsupportedCase('This original Neg/Sub composition is validated only for alpha=1')
 x,y=case.inputs;shape=list(torch.broadcast_shapes(x.shape,y.shape));dt=str(x.dtype).split('.')[-1]
 neg=make(SimpleNamespace(params={'input':[shape,dt],'output':[shape,dt]}),'NegFwdOp')
 sub=make(SimpleNamespace(params={'input':[shape,dt],'other':[shape,dt],'output':[shape,dt]}),'SubFwdOp')
 def run(x,y):return sub.implementation.run(x.expand(shape).contiguous(),neg.implementation.run(y.expand(shape).contiguous()))
 def cleanup():
  for c in (neg,sub):
   if c.cleanup:c.cleanup()
 return Candidate(Implementation(run),'ops-math::original Neg + Sub','Original public API composition x-(-y), alpha=1; broadcast materialized per call; all outputs checked by original checker.',{**neg.source_sha256,**sub.source_sha256,**source_hashes(Path(__file__))},cleanup=cleanup)

def install():
 from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
 NATIVE_CANDIDATES['AddFwdOp']=add
