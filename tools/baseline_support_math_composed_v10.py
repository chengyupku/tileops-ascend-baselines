"""Public original API compositions; unchanged native kernels and checker."""
from pathlib import Path
from types import SimpleNamespace
from functools import partial

def make(case,op):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,source_hashes
 from benchmarks.ascend_all_pairs.ops_math.graph import make as graph
 from benchmarks.ascend_all_pairs.ops_math.validated_abs.graph import make as abs_graph
 p=case.params;shape=p['x'][0];outshape,outdtype=p['output'];parts=[]
 def add(name,params,fn=graph):
  c=fn(SimpleNamespace(params=params),name);parts.append(c);return c.implementation.run
 def unary(name,s,fn=graph):return add(name,{'input':[s,'float32'],'output':[s,'float32']},fn)
 if op=='VectorNormFwdOp':
  order=float(p['ord'])
  absolute=unary('AbsFwdOp',shape,abs_graph)
  reduce=add('AmaxFwdOp' if order==float('inf') else 'SumFwdOp',{'x':[shape,'float32'],'output':[outshape,'float32'],'dim':p.get('dim'),'keepdim':p.get('keepdim',False)})
  if order==2:
   mul=add('MulFwdOp',{'input':[shape,'float32'],'other':[shape,'float32'],'output':[shape,'float32']});sqrt=unary('SqrtFwdOp',outshape)
  if order not in (1,2,float('inf')):raise ValueError('Unsupported norm order')
  def run(x):
   v=x.float()
   y=sqrt(reduce(mul(v,v))) if order==2 else reduce(absolute(v))
   return y.to(getattr(torch,outdtype))
 elif op=='CumprodFwdOp':
  axis=p.get('dim',-1)%len(shape);steps=[];offset=1
  while offset<shape[axis]:
   s=list(shape);s[axis]-=offset
   f=add('MulFwdOp',{'input':[s,'float32'],'other':[s,'float32'],'output':[s,'float32']});steps.append((offset,f));offset*=2
  def run(x):
   v=x.float()
   for offset,f in steps:
    head=v.narrow(axis,0,offset);left=v.narrow(axis,0,v.shape[axis]-offset);right=v.narrow(axis,offset,v.shape[axis]-offset)
    v=torch.cat((head,f(left,right)),dim=axis)
   return v.to(getattr(torch,outdtype))
 elif op=='BatchNormFwdOp':
  from benchmarks.ascend_all_pairs.ops_math.layernorm import make as norm
  c=norm(case,op);parts.append(c)
  def run(*args):return c.implementation.run(*args)[0]
 elif op=='CountNonzeroFwdOp':
  summation=add('SumFwdOp',{'x':[shape,'float32'],'output':[outshape,'float32'],'dim':p.get('dim'),'keepdim':False})
  def run(x):return summation(x.to(torch.bool).float()).to(torch.int64)
 else:raise ValueError(op)
 sources=source_hashes(Path(__file__))
 for part in parts:sources.update(part.source_sha256)
 def cleanup():
  for part in parts:
   if part.cleanup:part.cleanup()
 return Candidate(Implementation(run),'original ops-math public composition '+op,'Original native API composition, dtype/layout and return contract only.',sources,cleanup=cleanup)

def install():
 from benchmarks.ascend_all_pairs.ops_math import NATIVE_CANDIDATES
 from tools.baseline_support_math_logical_v8 import install as logical
 logical()
 for op in ('VectorNormFwdOp','CumprodFwdOp','BatchNormFwdOp','CountNonzeroFwdOp'):NATIVE_CANDIDATES[op]=partial(make,op=op)
