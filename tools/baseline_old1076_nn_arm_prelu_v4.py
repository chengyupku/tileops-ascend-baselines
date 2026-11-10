"""Expose channel weights as explicit broadcast dimensions to original PRelu."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1]
# Reuse unchanged deployment setup and candidate factory; insert only a public
# input-layout adapter before the existing measurement entry executes.
source=(ROOT/'tools/baseline_old1076_nn_arm_staging_v3.py').read_text()
insert='''
def prelu_channel(case):
 import torch
 from benchmarks.ascend_all_pairs.ops_nn.common import candidate,new_binding
 binding=new_binding()
 def run(x,weight):
  if weight.numel()==1: shaped=weight.reshape(*([1]*x.ndim))
  else:
   if x.ndim<2 or weight.numel()!=x.shape[1]:raise ValueError('Channel weight contract mismatch')
   shaped=weight.reshape(1,x.shape[1],*([1]*(x.ndim-2)))
  out=torch.empty_like(x)
  binding.call('PRelu',[x.contiguous(),shaped.contiguous()],[out],{})
  return out
 return candidate(binding,run,'PRelu','Reshape original channel weights [C] to [1,C,1,...] for original BroadcastBaseTiling; values, native PRelu and full checker unchanged.','activation/p_relu')
staging.CANDIDATES['PreluFwdOp']=prelu_channel
'''
source=source.replace('registry.candidate_factory=factory',insert+'\nregistry.candidate_factory=factory')
exec(compile(source,str(Path(__file__)), 'exec'))
