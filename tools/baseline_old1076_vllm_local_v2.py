"""Bind the original vLLM complete recurrence to the local original runtime."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'TileOPs'),str(ROOT/'TileOPs/src'),str(ROOT)]
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_all_pairs.vllm.linear_attention import linear_attention
def factory(case,library):
    assert library=='vllm-ascend' and case._call.signature.name=='GDNFwdOp'
    return lambda:linear_attention(case)
registry.candidate_factory=factory
runpy.run_path(str(ROOT/'tools/check_baseline_old1076_pair.py'),run_name='__main__')
