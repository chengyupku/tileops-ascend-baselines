"""One original AddLayerNorm API; descriptor and output binding only."""
from pathlib import Path

def build(case):
 import torch
 from nn_graph import GraphCall
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,source_hashes
 assert case._call.signature.name=='FusedAddLayerNormFwdOp'
 b=GraphCall('ops-nn');eps=float(case.params['eps'])
 def run(x,residual,w,bias):
  out=torch.empty_like(x);summed=torch.empty_like(x)
  stat=torch.empty((*x.shape[:-1],1),device=x.device,dtype=torch.float32)
  b.call('AddLayerNorm',[x.contiguous(),residual.contiguous(),w.contiguous(),bias.contiguous(),None],
         [out,stat,torch.empty_like(stat),summed],{'epsilon':eps,'additional_output':True})
  return out,summed
 root=Path(__file__).resolve().parent
 files=[Path(__file__),root/'nn_graph.py']
 for base in [b.vendor,root/'original-build/ops-nn/norm/add_layer_norm']:
  files.extend(f for f in base.rglob('*') if f.is_file() and f.suffix in ('.cpp','.h','.py','.json','.so','.o'))
 return Candidate(Implementation(run),'ops-nn::AddLayerNorm original graph JIT','Original API, unchanged inputs, original epsilon and both outputs; no arithmetic outside the API.',source_hashes(*files),cleanup=b.cleanup)
