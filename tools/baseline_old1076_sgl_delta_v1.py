"""Top-level DeltaNet binding: identity log-gate into original full SGL GDN API."""
import dataclasses,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.sgl.linear import _triton_gdn
from benchmarks.ascend_baselines import source_hashes

def factory(case,library):
    if library!='sgl-kernel-npu' or case._call.signature.name!='DeltaNetFwdOp':raise ValueError('DeltaNetFwdOp only')
    def build():
        import torch
        c=_triton_gdn(case)
        native=c.implementation
        def run(q,k,v,beta,initial,cu,cu_cpu):
            gate=torch.zeros(beta.shape,device=beta.device,dtype=torch.float32)
            return native.run(q,k,v,gate,beta.float(),initial,cu,cu_cpu,None,None)
        return dataclasses.replace(c,implementation=dataclasses.replace(native,run=run),source_sha256={**c.source_sha256,**source_hashes(Path(__file__))},adaptation=c.adaptation+' DeltaNet is log-gate zero (identity decay); preserve beta, scale, sequence and state; original full checker required.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/measure_baseline_cache_pair.py'),run_name='__main__')
