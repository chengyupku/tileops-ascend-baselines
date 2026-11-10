"""Bounded second interface attempt: partition independent packed sequences only."""
import dataclasses,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.sgl.linear import _triton_gdn
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
    if (library,case._call.signature.name)!=('sgl-kernel-npu','GDNFwdOp'):raise ValueError('Only SGL GDN interface')
    def build():
        import torch
        c=_triton_gdn(case);native=c.implementation
        def run(q,k,v,g,beta,initial,cu,cu_cpu,a_log,dt_bias):
            if cu_cpu is None:raise ValueError('Explicit CPU sequence boundaries required')
            bounds=cu_cpu.tolist()
            if q.shape[0]!=1 or bounds[0]!=0 or bounds[-1]!=q.shape[1]:raise ValueError('Expected packed independent sequences')
            output=torch.empty_like(v);states=[]
            for first in range(0,len(bounds)-1,4):
                last=min(first+4,len(bounds)-1);start,stop=bounds[first],bounds[last]
                local_cpu=torch.tensor([z-start for z in bounds[first:last+1]],dtype=cu_cpu.dtype)
                local=local_cpu.to(q.device)
                out,state=native.run(q[:,start:stop].contiguous(),k[:,start:stop].contiguous(),v[:,start:stop].contiguous(),g[:,start:stop].float().contiguous(),beta[:,start:stop].float().contiguous(),None if initial is None else initial[first:last].contiguous(),local,local_cpu,a_log,dt_bias)
                output[:,start:stop].copy_(out);states.append(state)
            return output,torch.cat(states,dim=0)
        return dataclasses.replace(c,implementation=dataclasses.replace(native,run=run),adaptation=c.adaptation+' Partition independent packed sequences in groups of4; preserve each full sequence and exact initial state; lossless gate/betaFP32 transport; copies included in timing.',source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
