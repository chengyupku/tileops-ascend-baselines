"""Original complete routing weighted sum with token/expert index binding."""
from pathlib import Path
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def post(case):
 import torch
 try:from .integration700_transformer_dense_graph import GraphCall
 except ImportError:from integration700_transformer_dense_graph import GraphCall
 from benchmarks.ascend_baselines import Candidate,source_hashes,UnsupportedCase
 from benchmarks.api import Implementation
 if case.params.get('epilogue') is not None or case.params.get('out_dtype') not in (None,str(case.inputs[0].dtype).removeprefix('torch.')):raise UnsupportedCase('Original routing output dtype/epilogue mismatch')
 binding=GraphCall('ops-transformer');vendor=P/'common/allpairs_transformer'
 def run(expert_output,topk_weights,inverse_indices):
  out=torch.empty((topk_weights.shape[0],expert_output.shape[-1]),dtype=expert_output.dtype,device=expert_output.device)
  binding.call('MoeFinalizeRoutingV2',[expert_output.reshape(-1,expert_output.shape[-1]).contiguous(),inverse_indices.reshape(topk_weights.shape).T.contiguous().flatten().int(),None,None,None,topk_weights.contiguous(),None],[out],{'drop_pad_mode':0,'k':topk_weights.shape[1]})
  return out
 files=[Path(__file__)]+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h','.so')]
 return Candidate(Implementation(run),'ops-transformer::MoeFinalizeRoutingV2','Single complete original native weighted routing; pure inverse-index layout transpose and output allocation.',source_hashes(*files),cleanup=binding.cleanup)
