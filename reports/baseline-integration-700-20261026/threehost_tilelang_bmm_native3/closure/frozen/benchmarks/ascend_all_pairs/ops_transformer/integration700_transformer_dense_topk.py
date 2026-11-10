"""Complete original MoeGatingTopKV2 with lossless logits widening."""
from pathlib import Path
import ctypes as C
P=Path('/opt/tileops-all-pairs/workspace/reports/baseline-integration-700-20261026/transformer_dense')
def topk(case):
    import torch
    try:from .integration700_transformer_dense_graph import GraphCall
    except ImportError:from integration700_transformer_dense_graph import GraphCall
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from benchmarks.api import Implementation
    vendor=P/'selection/selectionpairs_transformer'
    b=GraphCall('ops-transformer');p=case.params
    def run(gating_output,correction_bias=None):
        x=gating_output.float().contiguous();bias=None if correction_bias is None else correction_bias.float().contiguous();out=torch.empty((x.shape[0],p['top_k']),dtype=torch.float32,device=x.device);ids=torch.empty_like(out,dtype=torch.int32);full=torch.empty_like(x)
        attrs={'k':p['top_k'],'k_group':p['topk_group'],'group_count':p['num_expert_group'],'group_select_mode':1 if bias is not None else 0,'renorm':int(p['renormalize']),'norm_type':0 if p['scoring_func']=='softmax' else 1,'out_flag':False,'routed_scaling_factor':float(p['routed_scaling_factor']),'eps':0.}
        b.call('MoeGatingTopK',[x,bias,None,None],[out,ids,full],attrs,names=['x','bias','input_ids','tid2eid','y','expert_idx','out']);return out,ids
    files=[Path(__file__),vendor/'op_api/lib/libcust_opapi.so']+[f for f in vendor.rglob('*') if f.is_file() and f.suffix in ('.py','.cpp','.h')]
    return Candidate(Implementation(run),'ops-transformer::aclnnMoeGatingTopKV2','Complete native softmax/topk/renormalization; lossless FP32 input binding, full FP32 weights and int32 expert indices.',source_hashes(*files),cleanup=b.cleanup)
