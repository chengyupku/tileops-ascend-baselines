"""Complete original routing with lossless FP32 input binding and all3outputs."""
from pathlib import Path
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def pre(case):
 import torch
 try:from .integration700_transformer_dense_graph import GraphCall
 except ImportError:from integration700_transformer_dense_graph import GraphCall
 from benchmarks.ascend_baselines import Candidate,source_hashes,UnsupportedCase
 from benchmarks.api import Implementation
 if case.params['layout'].get('contiguous')!={'packing':'tight','metadata_kind':'physical_psum','alignment':1}:raise UnsupportedCase('Original complete tight routing layout required')
 binding=GraphCall('ops-transformer');experts=case.params['num_local_experts'];vendor=P/'pre/allpairs_transformer'
 def run(x,ids):
  routes=ids.numel();out=torch.empty((routes,x.shape[1]),dtype=torch.float32,device=x.device);inverse=torch.empty(routes,dtype=torch.int32,device=x.device);prefix=torch.empty(experts,dtype=torch.int32,device=x.device)
  binding.call('MoeInitRoutingV2',[x.float().contiguous(),ids.contiguous()],[out,inverse,prefix],{'active_num':routes,'expert_capacity':0,'expert_num':experts,'drop_pad_mode':0,'expert_tokens_count_or_cumsum_flag':1,'expert_tokens_before_capacity_flag':False})
  return out.to(x.dtype),prefix,inverse
 files=[Path(__file__)]+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h','.so')]
 return Candidate(Implementation(run),'ops-transformer::MoeInitRoutingV2','Complete native routing with exact FP32 widening, output dtype restoration and complete routing/prefix/inverse contract.',source_hashes(*files),cleanup=binding.cleanup)
