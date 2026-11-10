"""Original public FP32 product and full reduction APIs for post-permute."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry

def factory(case,library):
    def build():
        import torch
        from benchmarks.ascend_all_pairs.tilekernels.extended import native,result
        from benchmarks.ascend_baselines import UnsupportedCase
        if library!='TileKernels' or case._call.signature.name!='MoEPostPermuteFwdOp' or case.params['epilogue'] is not None:
            raise UnsupportedCase('Reviewed complete post-permute without epilogue only')
        multiply=native('mhc.norm_fn_kernel','mhc_fn_normw_merge_fwd')
        reduce=native('mhc.norm_fn_kernel','mhc_reduce_partials_and_rmsnorm_fwd')
        def run(expert_output,weights,inverse):
            tokens,topk=weights.shape
            hidden=expert_output.shape[-1]
            gathered=expert_output.reshape(-1,hidden)[inverse.long()].float().reshape(tokens,topk,hidden)
            factors=weights[:,:,None].expand(tokens,topk,hidden).contiguous().reshape(-1)
            left=gathered.contiguous().reshape(-1)
            count=left.numel()
            left=torch.nn.functional.pad(left,(0,-count%256))
            factors=torch.nn.functional.pad(factors,(0,-count%256))
            products=torch.empty_like(left).reshape(1,-1)
            multiply(left.reshape(1,-1),factors,products)
            rows=tokens*hidden
            width=(rows+23)//24
            values=products.reshape(-1)[:count].reshape(tokens,topk,hidden).permute(1,0,2).reshape(topk,rows)
            partials=torch.nn.functional.pad(values,(0,width*24-rows)).contiguous().reshape(topk,width,1,24)
            zeros=torch.zeros((topk,width,1),device=expert_output.device,dtype=torch.float32)
            summed=torch.empty((width,1,24),device=expert_output.device,dtype=torch.float32)
            sumsq=torch.empty((width,1),device=expert_output.device,dtype=torch.float32)
            normalized=torch.empty((width,24),device=expert_output.device,dtype=torch.float32)
            reduce(partials,zeros,summed,sumsq,normalized,1,1.,topk)
            dtype=expert_output.dtype if case.params['out_dtype'] is None else getattr(torch,case.params['out_dtype'])
            return summed.reshape(-1)[:rows].reshape(tokens,hidden).to(dtype)
        return result(run,'mhc.norm_fn_kernel original multiply + complete raw reduction',
                      'Gather by original inverse indices, lossless FP32 widening, original full product and complete top-k reduction APIs. Select raw sum before unused RMS output; final declared cast only.',
                      'mhc/norm_fn_kernel.py','mhc/norm_fn_asc.py')
    return build

registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
