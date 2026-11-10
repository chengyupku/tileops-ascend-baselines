"""Original SGL alltoall API with single-rank environment and metadata binding."""
from pathlib import Path
ROOT=Path('/root/dyq/workspace')
SOURCE=ROOT/'ascend_baselines/sgl-kernel-npu/python/deep_ep'
PRIVATE=Path(__file__).resolve().parent

def build(case):
    import sys, os
    import torch, torch_npu
    import torch.distributed as dist
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    sys.path[:0]=[str(PRIVATE/'hostbuild/build/deepep'),str(SOURCE)]
    from deep_ep.strategies.normal_strategy import AlltoAllNormalCommStrategy
    if not dist.is_initialized():
        dist.init_process_group('hccl',init_method='file://'+str(Path(os.environ['MOE_ATTEMPT'])/'comm_init'),rank=0,world_size=1)
    assert dist.get_world_size()==1
    strategy=AlltoAllNormalCommStrategy(None,dist.group.WORLD)
    p=case.params
    if len(case.inputs) == 2:
        def run(x, ids):
            nr,nrd,ne,mask,event=strategy.get_dispatch_layout(ids,p['num_local_experts'])
            rows,_,_,counts,h,_=strategy.dispatch(x,None,nr,nrd,mask,ne,ids,None,quant_mode='bf16')
            local=h['reversed_local_mapping'].reshape(-1).long()
            global_map=h['reversed_global_mapping']
            inverse=local if global_map is None else global_map[local]
            ends=torch.tensor(counts,device=x.device,dtype=torch.int32).cumsum(0).to(torch.int32)
            return rows,ends,inverse.to(torch.int32)
    else:
        def run(rows, weights, inverse):
            t,k=weights.shape; hdim=rows.shape[-1]
            # Route-row replication is layout binding; original combine does all arithmetic.
            routes=rows.reshape(-1,hdim)[inverse.long()].float().contiguous()
            mapping=torch.arange(t*k,device=rows.device,dtype=torch.int32)
            handle=dict(input_splits=[t*k],output_splits=[t*k],topk_weights=weights,reversed_local_mapping=mapping,reversed_global_mapping=None,hidden_shape=(t,hdim),hidden_shape_before_permute=(t,hdim),num_local_experts=1)
            result,_,_=strategy.combine(routes,handle)
            return result.to(rows.dtype)
    return Candidate(Implementation(run),'deep_ep.AlltoAllNormalCommStrategy.dispatch/combine','Original complete alltoall API, real single-rank HCCL; metadata/layout only.',source_hashes(Path(__file__),SOURCE/'deep_ep/strategies/normal_strategy.py'))
