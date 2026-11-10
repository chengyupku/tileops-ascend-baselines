"""Original graph APIs with public logical dtype and count composition adapters."""
from functools import partial
from pathlib import Path
from types import SimpleNamespace

def make(case,op):
 import torch
 from benchmarks.ascend_all_pairs.ops_math.graph import make as graph
 from benchmarks.ascend_baselines import Candidate,source_hashes
 from benchmarks.api import Implementation
 p=case.params;shape=p['x'][0];parts=[]
 if op in ('AllFwdOp','AnyFwdOp'):
  c=graph(SimpleNamespace(params={**p,'x':[shape,'bool']}),op);parts.append(c)
  def run(x):return c.implementation.run(x.to(torch.bool))
 else:
  c=graph(SimpleNamespace(params={'input':[shape,p['x'][1]],'other':[[] ,p['x'][1]],'output':[shape,'bool']}),'NeFwdOp');parts.append(c)
  summation=graph(SimpleNamespace(params={'x':[shape,'float32'],'output':[p['output'][0],'float32'],'dim':p.get('dim'), 'keepdim':False}),'SumFwdOp');parts.append(summation)
  def run(x):
   mask=c.implementation.run(x,torch.zeros((),device=x.device,dtype=x.dtype))
   return summation.implementation.run(mask.float()).to(torch.int64)
 def cleanup():
  for c in parts:
   if c.cleanup:c.cleanup()
 sources=source_hashes(Path(__file__))
 for c in parts:sources.update(c.source_sha256)
 return Candidate(Implementation(run),'original ops-math graph logical/counted predicate '+op,'Top-level boolean dtype conversion or original Ne + Sum composition; no kernel/compiler/checker changes.',sources,cleanup=cleanup)

def install():
 from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
 for op in ('AllFwdOp','AnyFwdOp','CountNonzeroFwdOp'):NATIVE_CANDIDATES[op]=partial(make,op=op)
