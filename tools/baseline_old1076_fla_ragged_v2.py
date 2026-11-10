"""Second interface attempt: preserve exact ragged spans and explicit state layout."""
import dataclasses,runpy,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.cache/old1076/fla'),str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.fla.providers import gdn
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
    if (library,case._call.signature.name)!=('flash-linear-attention-npu','GDNFwdOp'):raise ValueError('Only original FLA GDN')
    def build():
        import torch
        v_first=case.params.get('state_v_first',False)
        native_case=SimpleNamespace(params={**case.params,'state_v_first':False})
        c=gdn(native_case);impl=c.implementation
        def run(q,k,v,g,beta,initial,cu,cu_cpu,a_log,dt_bias):
            if cu_cpu is None or q.shape[0]!=1:raise ValueError('Packed sequences with explicit CPU offsets required')
            bounds=cu_cpu.tolist();out=torch.empty_like(v);states=[]
            for i,(start,stop) in enumerate(zip(bounds,bounds[1:])):
                if start==stop:raise ValueError('Empty span needs separate original API handling')
                state=None if initial is None else initial[i:i+1]
                if state is not None and v_first:state=state.transpose(-1,-2).contiguous()
                y,final=impl.run(q[:,start:stop].contiguous(),k[:,start:stop].contiguous(),v[:,start:stop].contiguous(),g[:,start:stop].float().contiguous(),beta[:,start:stop].float().contiguous(),state,None,None,a_log,dt_bias)
                out[:,start:stop].copy_(y);states.append(final.transpose(-1,-2).contiguous() if v_first else final)
            return out,torch.cat(states,dim=0)
        return dataclasses.replace(c,implementation=dataclasses.replace(impl,run=run),adaptation=c.adaptation+' Independent exact ragged spans without dense padding; explicit lossless state axis transpose at public API boundary.',source_sha256={**c.source_sha256,**source_hashes(Path(__file__))})
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
