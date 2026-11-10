"""Select original complete recurrent KDA through its public host API."""
import dataclasses
import importlib
import runpy
import sys
import types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
adapter=importlib.import_module('benchmarks.ascend_all_pairs.vllm.linear_attention')
from benchmarks.ascend_baselines import source_hashes
original_loader=adapter.kda_module
def load():
    module,sources=original_loader()
    def complete(q,k,v,g,beta,**kwargs):
        assert kwargs.pop('output_final_state',False)
        # The existing adapter allocates/clones this state privately every call.
        # Native inplace mode returns one complete final state per sequence.
        return module.fused_recurrent_kda(q,k,v,g,beta,inplace_final_state=True,
                                         sigmoid_beta=False,compute_gate=False,**kwargs)
    interface=types.SimpleNamespace(**module.__dict__)
    interface.chunk_kda=complete
    return interface,sources
adapter.kda_module=load
def factory(case,library):
    assert library=='vllm-ascend' and case._call.signature.name=='GDNFwdOp'
    def build():
        c=adapter.linear_attention(case)
        return dataclasses.replace(c,source_sha256=c.source_sha256|source_hashes(Path(__file__)),adaptation=c.adaptation+' Select complete original recurrent KDA public API for the full supplied sequences; use private cloned state and return full output/final state; no kernel/compiler/schedule changes.')
    return build
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
