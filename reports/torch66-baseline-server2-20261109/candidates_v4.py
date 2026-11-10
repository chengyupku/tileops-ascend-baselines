"""Native MoE routing and padding-safe chunk means; no custom kernels."""
import torch
from adapters.execution_boundary import ExecutionBoundary

def bind(case,args,previous):
    op=case._call.signature.name;p=case._call.params
    if op=='MoEPrePermuteFwdOp':
        import torch_npu
        experts=p['num_local_experts']
        assert args[1].numel()<2**24
        def run(hidden,ids):
            rows,inverse,counts,_=torch_npu.npu_moe_init_routing_v2(hidden,ids.to(torch.int32),expert_num=experts,expert_tokens_num_type=1,expert_tokens_num_flag=True,active_expert_range=[0,experts],quant_mode=-1,row_idx_type=0)
            ends=torch.cumsum(counts.float(),0).to(torch.int32)
            return rows,ends,inverse
        return ExecutionBoundary(run,tuple(args),lambda:None,{'implementation':'torch-npu eager','candidate':'v4','input_dependent_work_timed':True,'change':'native public MoE routing; exact FP32 count prefix for route totals below 2^24','targets':[1,2,4]})
    from candidates_v2 import bind as other
    return other(case,args,previous)
